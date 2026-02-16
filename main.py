from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
import httpx
from datetime import datetime
import json

app = FastAPI(title="Expense Tracker AI Assistant")

# CORS middleware for Django integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Django URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
CHROMA_PERSIST_DIR = "./chroma_db"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Global vector store
vector_store = None


class ChatRequest(BaseModel):
    question: str
    user_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


def call_lm_studio(prompt: str, context: str) -> str:
    """Call LM Studio API with context and question"""
    
    system_message = """You are a helpful financial assistant analyzing expense data. 
    Use the provided context to answer questions about income, expenses, and spending patterns.
    Be concise and specific with numbers. If asked about comparisons, calculate differences and percentages."""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
    ]
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                LM_STUDIO_URL,
                json={
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LM Studio error: {str(e)}")


def process_excel_data(df: pd.DataFrame) -> List[dict]:
    """Convert expense DataFrame to text chunks for RAG"""
    
    chunks = []
    
    # Group by month for monthly summaries
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Monthly summary chunks
    monthly_summary = df.groupby('month').agg({
        'price': ['sum', 'count'],
        'category': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    for _, row in monthly_summary.iterrows():
        month = str(row['month'])
        total = row['price']['sum']
        count = row['price']['count']
        categories = row['category']['<lambda>']
        
        chunk_text = f"""Month: {month}
        Total expenses: ${total:.2f}
        Number of transactions: {count}
        Categories: {json.dumps(categories, indent=2)}"""
        
        chunks.append({
            "text": chunk_text,
            "metadata": {"type": "monthly_summary", "month": month}
        })
    
    # Category summary chunks
    category_summary = df.groupby('category').agg({
        'price': ['sum', 'mean', 'count']
    }).reset_index()
    
    for _, row in category_summary.iterrows():
        category = row['category']
        total = row['price']['sum']
        avg = row['price']['mean']
        count = row['price']['count']
        
        chunk_text = f"""Category: {category}
Total spent: ${total:.2f}
Average transaction: ${avg:.2f}
Number of transactions: {count}"""
        
        chunks.append({
            "text": chunk_text,
            "metadata": {"type": "category_summary", "category": category}
        })
    
    # Individual transaction chunks (for recent/specific queries)
    for _, row in df.iterrows():
        chunk_text = f"""Transaction on {row['date'].strftime('%Y-%m-%d')}
Name: {row['name']}
Category: {row['category']}
Amount: ${row['price']:.2f}"""
        
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "type": "transaction",
                "date": row['date'].strftime('%Y-%m-%d'),
                "category": row['category']
            }
        })
    
    return chunks


@app.post("/upload_expenses")
async def upload_expenses(file: UploadFile = File(...), user_id: str = "default"):
    """Upload Excel file and create vector embeddings"""
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are supported")
    
    try:
        # Read Excel file
        df = pd.read_excel(file.file)
        
        # Validate required columns
        required_columns = ['name', 'price', 'category', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Process data into chunks
        chunks = process_excel_data(df)
        
        # Create collection name for user
        collection_name = f"expenses_{user_id.replace('-', '_')}"
        
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass
        
        # Create new collection
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"user_id": user_id}
        )
        
        # Add documents to collection
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"{user_id}_{i}" for i in range(len(chunks))]
        
        # Generate embeddings and add to collection
        embeddings_list = embeddings.embed_documents(texts)
        
        collection.add(
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "status": "success",
            "message": f"Uploaded {len(df)} transactions, created {len(chunks)} chunks",
            "user_id": user_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with AI assistant about expenses"""
    
    try:
        # Get user's collection
        collection_name = f"expenses_{request.user_id.replace('-', '_')}"
        
        try:
            collection = chroma_client.get_collection(collection_name)
        except:
            raise HTTPException(
                status_code=404, 
                detail="No expense data found. Please upload an Excel file first."
            )
        
        # Query vector store
        query_embedding = embeddings.embed_query(request.question)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        # Build context from retrieved documents
        context = "\n\n".join(results["documents"][0])
        sources = [f"{meta.get('type', 'unknown')}" for meta in results["metadatas"][0]]
        
        # Generate answer using LM Studio
        answer = call_lm_studio(request.question, context)
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():

    """Health check endpoint"""
    try:
        # Test LM Studio connection
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://10.0.85.2:1234/v1/models")
            lm_studio_status = response.status_code == 200
    except:
        lm_studio_status = False
    
    return {
        "status": "healthy",
        "lm_studio_connected": lm_studio_status
    }


@app.get("/")
async def root():
    return {
        "message": "Expense Tracker AI Assistant API",
        "endpoints": {
            "POST /upload_expenses": "Upload Excel file with expenses",
            "POST /chat": "Chat with AI about expenses",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
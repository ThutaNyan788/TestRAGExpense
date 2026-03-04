from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional, Dict
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from datetime import datetime
import json
import os

app = FastAPI(title="Expense Tracker AI Assistant")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "******")
GROQ_MODEL   = "llama-3.3-70b-versatile"          # fast & smart
CHROMA_PERSIST_DIR = "./chroma_db"

# Init Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Init embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Init Chroma
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    user_id: str
    conversation_history: Optional[List[Dict[str, str]]] = []   # NEW: multi-turn support


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    financial_insights: Optional[Dict] = None                   # NEW: structured insights


class SalaryRequest(BaseModel):
    user_id: str
    salary: float
    month: Optional[str] = None   # e.g. "2024-03"


# ─────────────────────────────────────────────
# Groq API call  (replaces LM Studio)
# ─────────────────────────────────────────────
def call_groq(
    question: str,
    context: str,
    conversation_history: List[Dict[str, str]] = [],
    financial_meta: Dict = {},
) -> str:
    """
    Call Groq with full conversation history + rich financial context.
    """

    # Build a richer system prompt using pre-computed financial meta
    salary_section = ""
    if financial_meta.get("salary"):
        salary    = financial_meta["salary"]
        total_exp = financial_meta.get("total_expenses", 0)
        balance   = salary - total_exp
        savings_pct = (balance / salary * 100) if salary else 0
        salary_section = f"""
SALARY & BUDGET OVERVIEW
Monthly salary: ${salary:,.2f}
Total expenses this period: ${total_exp:,.2f}
Remaining balance: ${balance:,.2f}
Savings rate: {savings_pct:.1f}%
"""

    system_message = f"""You are an expert personal finance assistant with deep knowledge of budgeting, spending analysis, and financial planning.

Your capabilities:
- Analyse expense data by category, month, merchant, and trend
- Compare spending vs salary / budget
- Detect unusual spending spikes or patterns
- Suggest actionable savings tips
- Answer follow-up questions using conversation history

Rules:
- Always cite exact figures from the provided context
- When comparing months, calculate absolute and percentage differences
- If salary data is available, show how expenses relate to income
- Be concise yet thorough; use bullet points for lists
- If the data does not contain enough information, say so clearly
{salary_section}"""

    # Build messages: system + history + new user turn
    messages = [{"role": "system", "content": system_message}]

    for turn in conversation_history[-6:]:      # keep last 6 turns for context window
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": f"EXPENSE DATA CONTEXT:\n{context}\n\n---\nQUESTION: {question}",
    })

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.4,       # lower = more factual
            max_tokens=1024,
            top_p=0.9,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")


# ─────────────────────────────────────────────
# Advanced data processing
# ─────────────────────────────────────────────
def process_excel_data(df: pd.DataFrame) -> List[dict]:
    """
    Convert expense DataFrame into rich, prioritised text chunks for RAG.
    Now includes: trends, top merchants, salary comparison, anomaly hints.
    """
    chunks = []

    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    df["week"]  = df["date"].dt.to_period("W")

    total_expenses    = df["price"].sum()
    total_transactions = len(df)
    avg_transaction   = df["price"].mean()
    max_expense       = df.loc[df["price"].idxmax()]
    date_range = (
        f"{df['date'].min().strftime('%Y-%m-%d')} to "
        f"{df['date'].max().strftime('%Y-%m-%d')}"
    )

    # ── 1. OVERALL SUMMARY ──────────────────────────────────────────────────
    overall_summary = f"""OVERALL EXPENSE SUMMARY
Total spending (all time): ${total_expenses:,.2f}
Total transactions: {total_transactions}
Average transaction: ${avg_transaction:,.2f}
Largest single expense: ${max_expense['price']:,.2f} ({max_expense['name']} on {max_expense['date'].strftime('%Y-%m-%d')})
Date range covered: {date_range}
Number of months tracked: {df['month'].nunique()}"""
    chunks.append({"text": overall_summary, "metadata": {"type": "overall_summary", "priority": "high"}})

    # ── 2. CATEGORY SUMMARIES ────────────────────────────────────────────────
    cat_totals = df.groupby("category")["price"].sum().sort_values(ascending=False)
    for category, total in cat_totals.items():
        group   = df[df["category"] == category]
        count   = len(group)
        avg     = group["price"].mean()
        pct     = (total / total_expenses * 100) if total_expenses else 0

        # Top 3 merchants in this category
        top_merchants = (
            group.groupby("name")["price"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
        merchant_lines = "\n".join(
            [f"  - {name}: ${amt:,.2f}" for name, amt in top_merchants.items()]
        )

        chunk_text = f"""CATEGORY SUMMARY: {category}
Total spent: ${total:,.2f} ({pct:.1f}% of all expenses)
Transaction count: {count}
Average per transaction: ${avg:,.2f}
Top merchants / items:
{merchant_lines}"""
        chunks.append({
            "text": chunk_text,
            "metadata": {"type": "category_summary", "category": str(category), "priority": "high"},
        })

    # ── 3. MONTHLY SUMMARIES (with MoM comparison) ──────────────────────────
    monthly_totals = df.groupby("month")["price"].sum()

    for month_period, month_df in df.groupby("month"):
        month       = str(month_period)
        total       = month_df["price"].sum()
        count       = len(month_df)
        categories  = month_df.groupby("category")["price"].sum().sort_values(ascending=False)

        # Month-over-month change
        sorted_months = sorted(monthly_totals.index)
        idx = sorted_months.index(month_period)
        if idx > 0:
            prev_total = monthly_totals[sorted_months[idx - 1]]
            mom_change = total - prev_total
            mom_pct    = (mom_change / prev_total * 100) if prev_total else 0
            mom_line   = f"Change vs previous month: {'+' if mom_change >= 0 else ''}${mom_change:,.2f} ({mom_pct:+.1f}%)"
        else:
            mom_line = "Change vs previous month: N/A (first month)"

        cat_lines = "\n".join([f"  - {cat}: ${amt:,.2f}" for cat, amt in categories.items()])

        chunk_text = f"""MONTHLY SUMMARY: {month}
Total expenses: ${total:,.2f}
{mom_line}
Number of transactions: {count}
Category breakdown:
{cat_lines}"""
        chunks.append({
            "text": chunk_text,
            "metadata": {"type": "monthly_summary", "month": month, "priority": "medium"},
        })

    # ── 4. WEEKLY TREND (for recent patterns) ───────────────────────────────
    for week_period, week_df in df.groupby("week"):
        week  = str(week_period)
        total = week_df["price"].sum()
        count = len(week_df)
        chunk_text = f"""WEEKLY SUMMARY: {week}
Total: ${total:,.2f}
Transactions: {count}"""
        chunks.append({
            "text": chunk_text,
            "metadata": {"type": "weekly_summary", "week": week, "priority": "low"},
        })

    # ── 5. TOP SPENDING DAYS ─────────────────────────────────────────────────
    daily = df.groupby(df["date"].dt.date)["price"].sum().sort_values(ascending=False).head(5)
    top_days_lines = "\n".join([f"  - {day}: ${amt:,.2f}" for day, amt in daily.items()])
    chunks.append({
        "text": f"TOP 5 HIGHEST SPENDING DAYS\n{top_days_lines}",
        "metadata": {"type": "spending_pattern", "priority": "medium"},
    })

    # ── 6. INDIVIDUAL TRANSACTIONS (recent 30) ───────────────────────────────
    for _, row in df.nlargest(30, "date").iterrows():
        chunk_text = (
            f"Transaction: {row['name']} | "
            f"Category: {row['category']} | "
            f"Amount: ${row['price']:,.2f} | "
            f"Date: {row['date'].strftime('%Y-%m-%d')}"
        )
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "type": "transaction",
                "date": row["date"].strftime("%Y-%m-%d"),
                "category": str(row["category"]),
                "amount": float(row["price"]),
                "priority": "low",
            },
        })

    return chunks


def compute_financial_meta(df: pd.DataFrame, salary: float = 0) -> Dict:
    """Pre-compute key financial figures to inject into the system prompt."""
    df["date"] = pd.to_datetime(df["date"])
    total_expenses = df["price"].sum()
    by_category    = df.groupby("category")["price"].sum().to_dict()
    return {
        "salary": salary,
        "total_expenses": total_expenses,
        "by_category": by_category,
    }


# ─────────────────────────────────────────────
# In-memory salary store (replace with DB in prod)
# ─────────────────────────────────────────────
salary_store: Dict[str, float] = {}


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.post("/upload_expenses")
async def upload_expenses(file: UploadFile = File(...), user_id: str = "default"):
    """Upload Excel file and create vector embeddings."""

    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files are supported.")

    try:
        df = pd.read_excel(file.file)

        required_columns = ["name", "price", "category", "date"]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        chunks          = process_excel_data(df)
        collection_name = f"expenses_{user_id.replace('-', '_')}"

        # Re-create collection cleanly
        try:
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"user_id": user_id},
        )

        texts      = [c["text"]     for c in chunks]
        metadatas  = [c["metadata"] for c in chunks]
        ids        = [f"{user_id}_{i}" for i in range(len(chunks))]
        emb_list   = embeddings.embed_documents(texts)

        collection.add(documents=texts, embeddings=emb_list, metadatas=metadatas, ids=ids)

        return {
            "status": "success",
            "message": f"Uploaded {len(df)} transactions → {len(chunks)} chunks indexed",
            "user_id": user_id,
            "collection_name": collection_name,
            "summary": {
                "total_expenses": float(df["price"].sum()),
                "months_covered": df["date"].nunique(),
                "categories": df["category"].nunique(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_salary")
async def set_salary(request: SalaryRequest):
    """Store monthly salary for a user so the AI can compare income vs expenses."""
    salary_store[request.user_id] = request.salary
    return {
        "status": "success",
        "user_id": request.user_id,
        "salary": request.salary,
        "message": "Salary saved. The AI will now provide income-aware analysis.",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Multi-turn chat with Groq LLM about expenses + salary.
    Supports conversation history for follow-up questions.
    """
    try:
        collection_name = f"expenses_{request.user_id.replace('-', '_')}"

        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception:
            raise HTTPException(
                status_code=404,
                detail="No expense data found. Please upload an Excel file first.",
            )

        question_lower = request.question.lower()

        # ── Smart retrieval strategy ─────────────────────────────────────
        is_summary_query = any(
            w in question_lower
            for w in ["total", "all", "overall", "how much", "sum", "spent", "expense"]
        )
        is_salary_query  = any(
            w in question_lower
            for w in ["salary", "income", "saving", "balance", "left", "remaining", "afford"]
        )
        is_trend_query   = any(
            w in question_lower
            for w in ["trend", "increase", "decrease", "compare", "month", "weekly", "pattern"]
        )

        n_results = 20 if (is_summary_query or is_salary_query) else 12

        # Build where filter
        priority_filter = None
        if is_summary_query or is_salary_query:
            priority_filter = {"priority": {"$in": ["high", "medium"]}}
        elif is_trend_query:
            priority_filter = {"priority": {"$in": ["high", "medium", "low"]}}

        query_emb = embeddings.embed_query(request.question)

        try:
            if priority_filter:
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=n_results,
                    where=priority_filter,
                )
                if not results["documents"][0]:
                    raise ValueError("Empty filtered results")
            else:
                raise ValueError("No filter — go to fallback")
        except Exception:
            results = collection.query(query_embeddings=[query_emb], n_results=n_results)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        docs_with_meta = sorted(
            zip(results["documents"][0], results["metadatas"][0]),
            key=lambda x: priority_order.get(x[1].get("priority", "low"), 3),
        )

        context = "\n\n".join(
            [f"--- [{meta.get('type','?').upper()}] ---\n{doc}" for doc, meta in docs_with_meta[:12]]
        )

        # ── Financial meta for system prompt ─────────────────────────────
        salary = salary_store.get(request.user_id, 0)
        all_docs = collection.get()
        # Reconstruct a lightweight df from stored overall_summary chunk
        financial_meta = {"salary": salary, "total_expenses": 0}
        for doc in all_docs["documents"]:
            if "Total spending (all time):" in doc:
                try:
                    line  = [l for l in doc.split("\n") if "Total spending (all time):" in l][0]
                    total = float(line.split("$")[1].replace(",", ""))
                    financial_meta["total_expenses"] = total
                except Exception:
                    pass
                break

        # ── Call Groq ────────────────────────────────────────────────────
        answer = call_groq(
            question=request.question,
            context=context,
            conversation_history=request.conversation_history or [],
            financial_meta=financial_meta,
        )

        sources = list({meta.get("type", "unknown") for _, meta in docs_with_meta[:12]})

        # ── Optional structured insights for the frontend ─────────────────
        financial_insights = None
        if salary and financial_meta["total_expenses"]:
            balance     = salary - financial_meta["total_expenses"]
            savings_pct = balance / salary * 100
            financial_insights = {
                "salary": salary,
                "total_expenses": financial_meta["total_expenses"],
                "balance": balance,
                "savings_rate_pct": round(savings_pct, 1),
            }

        return ChatResponse(answer=answer, sources=sources, financial_insights=financial_insights)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/{user_id}")
async def get_summary(user_id: str):
    """Return a quick financial summary without a question."""
    collection_name = f"expenses_{user_id.replace('-', '_')}"
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        raise HTTPException(status_code=404, detail="No data found for this user.")

    # Pull overall_summary and category_summary chunks
    all_docs  = collection.get()
    summaries = [
        doc for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
        if meta.get("type") in ("overall_summary", "category_summary")
    ]
    salary = salary_store.get(user_id, 0)
    context = "\n\n".join(summaries)

    answer = call_groq(
        question="Give me a comprehensive financial health report based on this data.",
        context=context,
        financial_meta={"salary": salary},
    )
    return {"report": answer, "salary_on_file": salary}


@app.get("/debug/collection/{user_id}")
async def debug_collection(user_id: str):
    collection_name = f"expenses_{user_id.replace('-', '_')}"
    try:
        collection = chroma_client.get_collection(collection_name)
        all_data   = collection.get()
        return {
            "collection_name": collection_name,
            "total_documents": collection.count(),
            "sample_documents": all_data["documents"][:3],
            "sample_metadata": all_data["metadatas"][:3],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """Health check — verifies Groq is reachable."""
    try:
        groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        groq_ok = True
    except Exception:
        groq_ok = False

    return {"status": "healthy", "groq_connected": groq_ok, "model": GROQ_MODEL}


@app.get("/")
async def root():
    return {
        "message": "Expense Tracker AI Assistant (Groq-powered)",
        "endpoints": {
            "POST /upload_expenses":     "Upload Excel expense file",
            "POST /set_salary":          "Set monthly salary for income analysis",
            "POST /chat":                "Multi-turn chat with AI about your finances",
            "GET  /summary/{user_id}":   "Get a full financial health report",
            "GET  /health":              "Health check",
            "GET  /debug/collection/id": "Debug vector store",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
# TestRAGExpense

An intelligent **Expense Tracker AI Assistant** API built with FastAPI, ChromaDB, and Groq's LLM. Query your expenses using natural language with a Retrieval-Augmented Generation (RAG) pipeline.

## Features

✨ **Core Capabilities**
- Upload Excel files containing expense data
- Automatic semantic chunking (overall summaries, monthly breakdowns, categories, recent transactions)
- Vector embeddings stored in persistent ChromaDB
- Natural language Q&A about spending patterns
- Multi-turn conversation support
- Powered by Groq's fast LLM inference

🛠️ **Technical Features**
- FastAPI REST API with auto-generated docs
- Excel validation with required schema enforcement
- HuggingFace embeddings for semantic search
- ChromaDB for persistent vector storage
- CORS middleware for cross-origin requests
- Health check endpoint

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Data Processing**: Pandas, OpenPyXL
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **HTTP**: HTTPX

## Project Structure

```
main.py                         # FastAPI app and RAG logic
requirements.txt                # Python dependencies
test_lm_studio.py               # Basic LM Studio API test
test_lm_studio_connection.py    # LM Studio connection diagnostics
chroma_db/                      # Persistent ChromaDB data
```

## Installation

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd TestRAGExpense
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the server

```bash
uvicorn main:app --reload --port 8001
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from [https://console.groq.com](https://console.groq.com)

## Configuration

Key settings in `main.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `GROQ_API_KEY` | Environment variable | Groq API authentication |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model for response generation |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector database storage location |

## Run the API

```bash
python main.py
```

Server starts at **`http://0.0.0.0:8001`**

**Interactive API docs**: `http://localhost:8001/docs`

## Expected Excel Format

Your uploaded Excel file must include these required columns:

| Column | Type | Example |
|--------|------|---------|
| `name` | string | "Coffee", "Laptop" |
| `price` | float | 4.50, 1299.99 |
| `category` | string | "Food", "Electronics" |
| `date` | date | 2026-01-03 |

### Example Data

```
name         | price  | category    | date
-------------|--------|-------------|----------
Coffee       | 4.50   | Food        | 2026-01-03
Taxi         | 12.00  | Transport   | 2026-01-03
Groceries    | 48.90  | Food        | 2026-01-04
Laptop       | 1299.99| Electronics | 2026-01-05
```

**Supported formats**: `.xlsx`, `.xls`

## API Endpoints

### `GET /`
Returns API intro and available endpoints.

### `GET /health`
Returns service status and LM Studio connectivity check.

### `POST /upload_expenses`
Uploads an Excel file and builds embeddings for a user.

Query parameter:
- `user_id` (optional, default: `default`)

Form data:
- `file`: Excel file

Example:

```bash
curl -X POST "http://localhost:8001/upload_expenses?user_id=u1" \
	-F "file=@expenses.xlsx"
```

### `POST /chat`
Ask questions about a user’s uploaded expenses.

Request body:

```json
{
	"question": "How much did I spend in total?",
	"user_id": "u1"
}
```

Example:

```bash
curl -X POST "http://localhost:8001/chat" \
	-H "Content-Type: application/json" \
	-d '{"question":"How much did I spend on food?","user_id":"u1"}'
```

### `GET /debug/collection/{user_id}`
Debug endpoint to inspect stored documents and metadata in ChromaDB.

## How It Works (RAG Flow)

1. Upload Excel data.
2. Data is transformed into text chunks:
	 - Overall summary
	 - Category summaries
	 - Monthly summaries
	 - Recent transactions
3. Chunks are embedded and stored in user-specific Chroma collections.
4. On chat request, relevant chunks are retrieved.
5. Retrieved context + user question are sent to LM Studio for final answer generation.

## LM Studio Testing Scripts

Run these before starting the API if needed:

```bash
python test_lm_studio_connection.py
python test_lm_studio.py
```

## Notes

- CORS is currently open (`allow_origins=["*"]`) for development.
- For production, restrict CORS origins and secure your LM Studio endpoint.
- Chroma data persists in `chroma_db/`.
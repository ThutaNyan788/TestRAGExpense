# TestRAGExpense

An Expense Tracker AI Assistant API built with FastAPI + RAG.

This project lets you:
- Upload an Excel file of expenses.
- Convert the data into semantic chunks (overall, monthly, category, and recent transactions).
- Store embeddings in ChromaDB.
- Ask natural-language questions about spending through a chat endpoint.

The app is designed to work with an LM Studio-compatible chat completion endpoint.

## Features

- FastAPI backend with REST endpoints.
- Excel ingestion (`.xlsx` / `.xls`) with required schema validation.
- RAG pipeline using:
	- `sentence-transformers/all-MiniLM-L6-v2` embeddings
	- ChromaDB persistent collections
- Smart retrieval behavior for summary/total-style questions.
- Health endpoint to check LM Studio connectivity.

## Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- Pandas + OpenPyXL
- ChromaDB
- LangChain Community Embeddings
- HTTPX

## Project Structure

```
main.py                         # FastAPI app and RAG logic
requirements.txt                # Python dependencies
test_lm_studio.py               # Basic LM Studio API test
test_lm_studio_connection.py    # LM Studio connection diagnostics
chroma_db/                      # Persistent ChromaDB data
```

## Installation

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd TestRAGExpense
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

In `main.py`, these constants are used:

- `LM_STUDIO_URL`: URL for chat completions endpoint.
- `CHROMA_PERSIST_DIR`: directory where Chroma stores vectors (`./chroma_db`).

If your LM Studio server is local, you may want to change it to something like:

```python
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
```

## Run the API

```bash
python main.py
```

Server starts at:

- `http://0.0.0.0:8001`

Interactive docs:

- `http://localhost:8001/docs`

## Expected Excel Format

Your uploaded Excel file must include these columns:

- `name`
- `price`
- `category`
- `date`

Example rows:

| name      | price | category  | date       |
|-----------|-------|-----------|------------|
| Coffee    | 4.50  | Food      | 2026-01-03 |
| Taxi      | 12.00 | Transport | 2026-01-03 |
| Groceries | 48.90 | Food      | 2026-01-04 |

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
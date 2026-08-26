# GPT Clone (LangGraph + FastAPI)

A fast, containerized AI Chatbot with Memory, RAG (Document Search), and Web Search capabilities.

---

## ⚡ Quick Run

### Option 1: Local (`uv`)
```powershell
$env:PYTHONPATH="src"
uv run python src/app/app.py
```

### Option 2: Docker
```powershell
docker compose up -d --build
```
👉 **App URL:** `http://localhost:8080`

---

## 🔑 Required API Keys (`.env`)

Create a `.env` file with:

```env
OPENAI_API_KEY=your_key_here     # LLM & RAG Embeddings
TAVILY_API_KEY=your_key_here     # Live Web Search
GOOGLE_API_KEY=your_key_here     # Gemini Models (Optional)
GROQ_API_KEY=your_key_here       # Groq Models (Optional)
```

---

## 📦 Core Packages & Why

- **`langgraph`**: Agent state graph & tool orchestration logic.
- **`fastapi` + `uvicorn`**: Web backend with streaming SSE chat responses.
- **`chromadb`**: Vector database for RAG document search (PDF/DOCX).
- **`sqlalchemy`**: SQLite store for chat threads & long-term memory.
- **`langchain-tavily`**: Real-time web search capability.

import os
import certifi
from datetime import datetime
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.rag import embeddings

# Setup environment CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Ensure directories exist
Path("data").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)

# Initialize Chroma store for episodic memory
episodic_store = Chroma(
    collection_name="agentic_episodic_memory",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


def save_episodic_memory(thread_id: str, summary: str):
    """
    Save or update the episodic summary of a conversation thread in the vector store.
    Uses the thread_id as the document ID to prevent duplicate summaries for the same thread.
    """
    try:
        episodic_store.delete(ids=[thread_id])
    except Exception:
        pass

    doc = Document(
        page_content=summary,
        metadata={
            "thread_id": thread_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "episodic"
        }
    )
    
    episodic_store.add_documents([doc], ids=[thread_id])


def search_episodic_memory(query: str, current_thread_id: str = None, limit: int = 3) -> str:
    """
    Query the episodic memory vector database to find semantically relevant past events/conversations.
    Excludes the current thread_id to prevent the agent from recalling its active conversation state as a "past event".
    """
    filter_dict = {"type": "episodic"}
    
    # Try using metadata filter. Chroma supports unequal $ne operator
    # If the vector store backend has issues with complex filters, we catch and fallback
    try:
        if current_thread_id:
            filter_dict = {
                "$and": [
                    {"type": "episodic"},
                    {"thread_id": {"$ne": current_thread_id}}
                ]
            }
        docs = episodic_store.similarity_search(query, k=limit, filter=filter_dict)
    except Exception as e:
        print(f"[Episodic Memory] Filtering failed with error: {e}. Falling back to manual Python filtering.")
        # Fallback: Retrieve more documents and filter out current_thread_id programmatically
        docs = episodic_store.similarity_search(query, k=limit + 5)
        if current_thread_id:
            docs = [d for d in docs if d.metadata.get("thread_id") != current_thread_id][:limit]

    if not docs:
        return ""

    results = []
    for doc in docs:
        ts = doc.metadata.get("timestamp", "")
        # Try formatting timestamp nicely
        try:
            dt = datetime.fromisoformat(ts)
            formatted_ts = dt.strftime("%b %d, %Y %I:%M %p")
        except Exception:
            formatted_ts = ts

        results.append(f"[{formatted_ts}] (Conversation {doc.metadata.get('thread_id', '')[:8]}): {doc.page_content}")

    return "\n".join(results)


def list_all_episodic_memories():
    """
    Get list of all stored episodic memories (conversation summaries) sorted by timestamp descending.
    """
    try:
        data = episodic_store.get()
        results = []
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        for i in range(len(ids)):
            meta = metadatas[i] or {}
            if meta.get("type") == "episodic":
                results.append({
                    "thread_id": meta.get("thread_id", ids[i]),
                    "summary": documents[i],
                    "timestamp": meta.get("timestamp", "")
                })
                
        # Sort by timestamp descending (newest memories first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results
    except Exception as e:
        print(f"[Episodic Memory] Error listing memories: {e}")
        return []


def delete_episodic_memory(thread_id: str) -> bool:
    """
    Remove the episodic memory of a thread from the vector database.
    """
    try:
        episodic_store.delete(ids=[thread_id])
        return True
    except Exception as e:
        print(f"[Episodic Memory] Error deleting memory for thread {thread_id}: {e}")
        return False


def update_episodic_memory_background(thread_id: str, model_name: str):
    """
    Background task to summarize the chat log of a thread and save it in episodic memory.
    Runs after a message is successfully saved.
    """
    from utils.db import get_chat_history
    from utils.agent import normalize_model_name
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    
    # 1. Fetch chat messages from database
    messages = get_chat_history(thread_id)
    if not messages or len(messages) < 2:
        return  # Thread is empty or too short for a summary (needs at least one human-assistant pair)

    # Format history as text for the LLM
    history_text = ""
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    # 2. Ask the model to summarize the conversation episode
    prompt = f"""You are a memory consolidation assistant.
Analyze the following conversation history between a User and an Assistant.
Generate a concise, factual summary of this conversation episode.
Focus on:
1. What the user requested or discussed.
2. The key information, solutions, code, or answers provided.
3. Important preferences, facts, or context established.

Keep the summary structured, factual, clear, and under 150 words.
Do not use the words "User" or "Assistant" in the summary if possible; write it as a concise historical record (e.g., "Discussed FastAPI configuration, set up SQLite, and resolved ca-certificates SSL verification issues").

Conversation History:
{history_text}

Summary:"""

    try:
        selected_model = normalize_model_name(model_name)
        
        if selected_model == "openai-4.1-nano":
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        elif selected_model == "llama-3.3-70b-versatile":
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model=selected_model,
                temperature=0.3
            )
            
        response = llm.invoke(prompt)
        summary = response.content.strip()
        
        if summary:
            save_episodic_memory(thread_id, summary)
            print(f"[Memory System] Episodic memory updated in background for thread: {thread_id}")
    except Exception as e:
        print(f"[Memory System] Failed to run background summarization for thread {thread_id}: {e}")

from utils.rag import retrieve_from_rag
from utils.db import search_memory
from utils.db import save_memory
import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch


load_dotenv(override=True)

CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id

web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth = "advanced"
)



@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"
    
    
@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )



@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )


@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search and retrieve information from uploaded documents.
    Use this whenever the user asks about uploaded files, documents, PDFs, DOCX, TXT, or notes 
    (e.g., "what does the document say", "summarize the file", "what is this file about", or specific questions about the document content).
    """

    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )


tools = [
    web_search,
    calculator,
    remember_this,
    recall_memory,
    search_uploaded_documents
]
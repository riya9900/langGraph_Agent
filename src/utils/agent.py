from langgraph.checkpoint.sqlite import SqliteSaver
import certifi
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from utils.tools import tools

import certifi

load_dotenv(override=True)

# Print OpenAI key ending for verification
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"[Agent Module] Loaded OpenAI key ending in: ...{openai_key[-8:]}")
else:
    print("[Agent Module] OpenAI key NOT found in environment!")

os.environ["SSL_CERTI_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

Path("data").mkdir(exist_ok=True)

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

DEFAULT_MODEL=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ALLOWED_MODELS = {
    "gemini-2.5-flash",
    "llama-3.3-70b-versatile",
    "openai-4.1-nano",
}

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, use web search.
- If the user asks about an uploaded document (including general questions like "what is in this document", "what does this doc tell", "summarize it", or "what is this file about"), you MUST call the search_uploaded_documents tool with a relevant search query to fetch the contents. Never claim you don't know or don't have access without calling the tool first.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""

def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name

def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected Gemini, OpenAI, or Groq model.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model == "openai-4.1-nano":
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            streaming=True,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif selected_model == "llama-3.3-70b-versatile":
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    else:
        # Initialize ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.3,
            streaming=True
        )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)

# Caching to save time to build once...

_AGENT_CACHE = {}

def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]

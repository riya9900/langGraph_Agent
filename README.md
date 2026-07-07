# gpt-clone-langGraph

A GPT Clone project implemented using LangGraph, FastAPI, and ChromaDB.

## Stack Overview
- **Orchestration / LLM Integration:** `langchain`, `langgraph`, `langchain-google-genai`
- **Backend Framework:** `fastapi`, `uvicorn`, `jinja2`
- **Database / Vector Store:** `chromadb`, `langchain-chroma`, `sqlalchemy`
- **Tooling / Search:** `langchain-tavily`, `tavily-python`
- **Document Loading:** `pypdf`, `docx2txt`

## Installation

Ensure you have [Poetry](https://python-poetry.org/) installed, then run:

```bash
poetry install
```

 $env:PYTHONPATH="src"
  poetry run python src/app/app.py

## Running the Application

Activate the virtual environment and start the development server:

```bash
poetry shell
$env:PYTHONPATH="src"
 poetry run python src/app/app.py
```

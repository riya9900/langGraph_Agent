from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import certifi
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import docx2txt

# Future imports for processing image, excel, docx, pptx, etc. using LlamaCloud
# from llama_cloud import LlamaCloud
# from llama_index.core.node_parser import SentenceSplitter




load_dotenv(override=True)

# Print OpenAI key ending for verification
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"[RAG Module] Loaded OpenAI key ending in: ...{openai_key[-8:]}")
else:
    print("[RAG Module] OpenAI key NOT found in environment!")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
vectorstore = Chroma(
    collection_name="agentic_chatbot_docs_small",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# Commented out local python-based extraction tools as requested:
# def read_file_text(file_path: str) -> str:
#     path = Path(file_path)
#     suffix = path.suffix.lower()
# 
#     if suffix == ".pdf":
#         reader = PdfReader(file_path)
#         text = ""
# 
#         for page in reader.pages:
#             text += page.extract_text() or ""
#             text += "\n"
# 
#         return text
# 
#     if suffix == ".docx":
#         return docx2txt.process(file_path)
# 
#     if suffix in [".txt", ".md", ".py", ".csv"]:
#         return path.read_text(encoding="utf-8", errors="ignore")
# 
#     raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")


def add_document_to_rag(file_path: str, thread_id: str):
    from llama_parse import LlamaParse
    
    # Initialize the LlamaCloud parser
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown"
    )
    
    # Parse document using LlamaCloud API
    llama_docs = parser.load_data(file_path)
    text = "\n\n".join([doc.text for doc in llama_docs])

    if not text.strip():
        raise ValueError("No text could be extracted from this file by LlamaCloud.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name
            }
        )
        for chunk in chunks
    ]

    vectorstore.add_documents(docs)

    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }





def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={"thread_id": thread_id}
    )

    if not docs:
        return "No relevant uploaded document content found."

    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(results)

"""Configuration settings for PDF RAG system."""

import os
from pathlib import Path

from dotenv import load_dotenv
from typing_extensions import TypedDict

load_dotenv()


class FileInfo(TypedDict):
    """File metadata for tracking changes."""

    modified_time: float
    size: int


class Config:
    """Configuration constants."""

    # Directories
    PDF_DIRECTORY = Path("./pdf_rag/pdfs")
    DB_LOCATION = Path("./pdf_rag/pdf_rag_agent/chroma_langchain_db")
    METADATA_FILE = Path("./pdf_rag/pdf_rag_agent/processed_files.json")

    # Text processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 20

    # API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    # Models
    LLM_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
        cls.DB_LOCATION.parent.mkdir(parents=True, exist_ok=True)
        cls.METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)


# Chat template
CHAT_TEMPLATE = """
You are an expert assistant that answers questions based on the provided
PDF documents.

Here are relevant excerpts from the documents: {context}

Question: {question}

Please provide a comprehensive answer based on the context above.
If the answer cannot be found in the provided context, please say so.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from rag.settings import DB_DIR, USE_OPENAI, EMBED_MODEL


def get_embeddings():
    if USE_OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBED_MODEL)
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vectorstore():
    return Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())


def query_top_k(question: str, k: int = 4) -> List[Document]:
    vs = get_vectorstore()
    return vs.similarity_search(question, k=k)

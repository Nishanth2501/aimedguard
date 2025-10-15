from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from rag.settings import PDF_DIR, DB_DIR, USE_OPENAI, EMBED_MODEL


def get_embeddings():
    if USE_OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBED_MODEL)
    # Fallback: sentence-transformers (downloads once)
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def run():
    assert PDF_DIR.exists(), f"Missing PDFs dir: {PDF_DIR}"
    DB_DIR.mkdir(parents=True, exist_ok=True)

    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(
        chunks, embedding=get_embeddings(), persist_directory=str(DB_DIR)
    )
    vs.persist()
    print(f"Success: Ingested {len(chunks)} chunks into {DB_DIR}")


if __name__ == "__main__":
    run()

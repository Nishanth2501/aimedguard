from __future__ import annotations
from pathlib import Path
import os

PDF_DIR = Path("data/raw/compliance")
DB_DIR = Path("rag/vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # OpenAI default
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

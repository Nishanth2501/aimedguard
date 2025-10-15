# data/etl/etl_compliance.py
from __future__ import annotations
from pathlib import Path
import re, time
import pandas as pd
import fitz  # PyMuPDF
from ._common import log_run

RAW_DIR = Path("data/raw/compliance")
OUT = Path("data/processed/compliance/policies_index.parquet")
LOG = Path(f"logs/policies_index_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

CHUNK_SIZE = 1000
OVERLAP = 150
KEYWORDS = ("HIPAA", "MEDICARE", "CMS", "PRIVACY", "CLAIMS")


def chunk_text(text: str, size=CHUNK_SIZE, overlap=OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        yield text[start:end], start // size
        start = max(end - overlap, end)


def index_pdf(path: Path):
    doc = fitz.open(path)
    rows = []
    for p in range(len(doc)):
        page = doc.load_page(p)
        txt = page.get_text("text")
        for chunk, ix in chunk_text(txt):
            kws = [k for k in KEYWORDS if k in chunk.upper()]
            rows.append(
                {
                    "doc_id": path.stem,
                    "file": path.name,
                    "page": p + 1,
                    "chunk_ix": ix,
                    "text": chunk,
                    "keywords_found": ",".join(kws),
                }
            )
    return rows


def run():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        rows = index_pdf(pdf)
        all_rows.extend(rows)
        log_run(
            LOG,
            {
                "file": pdf.name,
                "pages_indexed": len(set(r["page"] for r in rows)),
                "chunks": len(rows),
            },
        )
    df = pd.DataFrame(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    log_run(
        LOG,
        {
            "dataset": "policies_index",
            "status": "SUCCESS",
            "total_chunks": int(len(df)),
        },
    )


if __name__ == "__main__":
    run()

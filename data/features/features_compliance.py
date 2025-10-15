# data/features/features_compliance.py
from __future__ import annotations
import time, json, re
from pathlib import Path
import pandas as pd

RAW = Path("data/processed/compliance/policies_index.parquet")
OUT = Path("data/features/compliance_topics.parquet")
LOG = Path(f"logs/features_compliance_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

TOPICS = {
    "t_hipaa": r"\bhipaa\b",
    "t_privacy": r"\bprivacy|confidential\b",
    "t_billing": r"\bbill|billing|claims?\b",
    "t_lab": r"\blab|laborator(y|ies)\b",
    "t_preventive": r"\bpreventive|screening\b",
    "t_reimburse": r"\breimburse|allow(able|ed)\b",
}


def run():
    df = pd.read_parquet(RAW)
    if "text" not in df.columns:
        raise ValueError("policies_index.parquet must include 'text' column")

    # Topic tags as booleans
    for name, pat in TOPICS.items():
        df[name] = df["text"].str.contains(pat, flags=re.I, regex=True, na=False)

    # Rolling up to doc-level prevalence
    doc_topics = (
        df.groupby("doc_id")[[*TOPICS.keys()]].mean().add_suffix("_share").reset_index()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "dataset": "compliance_topics",
                    "rows": int(len(df)),
                    "topic_positive_counts": {
                        k: int(df[k].sum()) for k in TOPICS.keys()
                    },
                }
            )
            + "\n"
        )

    # (Optional) also write doc-level summary for quick joins if needed
    doc_out = OUT.with_name("compliance_topics_docs.parquet")
    doc_topics.to_parquet(doc_out, index=False)

    print(f"✅ compliance_topics saved → {OUT}  rows={len(df)}")
    print(f"✅ compliance_topics_docs saved → {doc_out}  rows={len(doc_topics)}")


if __name__ == "__main__":
    run()

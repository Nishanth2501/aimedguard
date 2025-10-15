# data/features/combine_features.py
from __future__ import annotations
import time, json
from pathlib import Path
import pandas as pd
import numpy as np

CLAIMS = Path("data/features/claims_features.parquet")
OPS = Path("data/features/operational_features.parquet")
POLICY = Path(
    "data/features/compliance_topics_docs.parquet"
)  # doc-level summary created above
OUT = Path("data/features/final_feature_mart.parquet")
LOG = Path(f"logs/features_combine_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def run():
    claims = pd.read_parquet(CLAIMS)  # key: npi
    ops = pd.read_parquet(OPS)  # key: facility_id (and state context)
    policy = pd.read_parquet(POLICY)  # doc_id summaries (global context)

    # Create a simple state context from claims by mapping most common state per npi (if available)
    # If your claims.parquet lacks provider->facility join, we join on state-level signals only.
    # Prepare a compact state signal from OPS (already merged with state complications)
    state_cols = [c for c in ops.columns if c.startswith("state_worse_")]
    ops_state = ops[["state"] + state_cols].drop_duplicates(subset=["state"])

    # Claims likely has 'state' at provider-level in raw rows; compute dominant state per NPI
    if "state" in claims.columns:
        dom_state = (
            claims.assign(_one=1)
            .groupby(["npi", "state"])["_one"]
            .sum()
            .reset_index()
            .sort_values(["npi", "_one"], ascending=[True, False])
            .drop_duplicates(subset=["npi"])
            .rename(columns={"state": "dom_state"})[["npi", "dom_state"]]
        )
        claims = claims.merge(dom_state, on="npi", how="left")
        # attach state context
        claims = claims.merge(
            ops_state,
            left_on="dom_state",
            right_on="state",
            how="left",
            suffixes=("", "_ctx"),
        )
        claims = claims.drop(columns=["state_ctx"], errors="ignore")

    # Add policy topic priors as global features (broadcast)
    for c in policy.columns:
        if c.endswith("_share"):
            claims[c] = policy[c].mean()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    claims.to_parquet(OUT, index=False)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "dataset": "final_feature_mart",
                    "rows": int(len(claims)),
                    "columns": list(claims.columns),
                }
            )
            + "\n"
        )
    print(f"✅ final_feature_mart saved → {OUT}  rows={len(claims)}")


if __name__ == "__main__":
    run()

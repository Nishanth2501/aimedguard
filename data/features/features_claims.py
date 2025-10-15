# data/features/features_claims.py
from __future__ import annotations
import time, json
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/processed/claims/claims.parquet")
OUT = Path("data/features/claims_features.parquet")
LOG = Path(f"logs/features_claims_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def run():
    df = pd.read_parquet(RAW)

    # Safety: ensure expected columns exist
    needed = {
        "npi",
        "state",
        "hcpcs",
        "pos",
        "total_services",
        "total_beneficiaries",
        "avg_submitted_charge",
        "avg_allowed_amount",
        "avg_payment_amount",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in claims.parquet: {missing}")

    # Aggregate per provider (NPI)
    g = (
        df.groupby("npi")
        .agg(
            total_services=("total_services", "sum"),
            total_beneficiaries=("total_beneficiaries", "sum"),
            mean_charge=("avg_submitted_charge", "mean"),
            mean_allowed=("avg_allowed_amount", "mean"),
            mean_payment=("avg_payment_amount", "mean"),
            distinct_hcpcs=("hcpcs", "nunique"),
            distinct_pos=("pos", "nunique"),
            states_seen=("state", "nunique"),
        )
        .reset_index()
    )

    # Ratios (handle div-by-zero safely)
    g["pay_ratio"] = np.where(
        g["mean_charge"] > 0, g["mean_payment"] / g["mean_charge"], np.nan
    )
    g["allow_ratio"] = np.where(
        g["mean_charge"] > 0, g["mean_allowed"] / g["mean_charge"], np.nan
    )
    g["svc_per_bene"] = np.where(
        g["total_beneficiaries"] > 0,
        g["total_services"] / g["total_beneficiaries"],
        np.nan,
    )

    # Provider type one-hots if available
    if "provider_type" in df.columns:
        ptype = pd.get_dummies(
            df[["npi", "provider_type"]].dropna().drop_duplicates(),
            columns=["provider_type"],
            prefix="ptype",
        )
        g = g.merge(ptype, on="npi", how="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT, index=False)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "dataset": "claims_features",
                    "rows": int(len(g)),
                    "columns": list(g.columns),
                }
            )
            + "\n"
        )
    print(f"✅ claims_features saved → {OUT}  rows={len(g)}")


if __name__ == "__main__":
    run()

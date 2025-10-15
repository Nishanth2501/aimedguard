# data/etl/etl_claims.py
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from ._common import (
    normalize_obj,
    to_zip5,
    cast_nullable_int,
    to_float,
    winsorize,
    nullify_strings,
    log_run,
)

RAW = Path("data/raw/claims/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv")
OUT = Path("data/processed/claims/claims.parquet")
LOG = Path(f"logs/claims_clean_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

RENAME = {
    "Rndrng_NPI": "npi",
    "Rndrng_Prvdr_State_Abrvtn": "state",
    "Rndrng_Prvdr_Zip5": "zip5",
    "Rndrng_Prvdr_Type": "provider_type",
    "Rndrng_Prvdr_Crdntls": "credentials",
    "HCPCS_Cd": "hcpcs",
    "HCPCS_Desc": "hcpcs_desc",
    "HCPCS_Drug_Ind": "hcpcs_drug_ind",
    "Place_Of_Srvc": "pos",
    "Tot_Benes": "total_beneficiaries",
    "Tot_Srvcs": "total_services",
    "Tot_Bene_Day_Srvcs": "total_bene_day_services",
    "Avg_Sbmtd_Chrg": "avg_submitted_charge",
    "Avg_Mdcr_Alowd_Amt": "avg_allowed_amount",
    "Avg_Mdcr_Pymt_Amt": "avg_payment_amount",
    "Avg_Mdcr_Stdzd_Amt": "avg_std_amount",
    "Rndrng_Prvdr_RUCA": "ruca_code",
    "Rndrng_Prvdr_RUCA_Desc": "ruca_desc",
}

KEEP = list(RENAME.values())


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    # keep modeling-useful columns only if present
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()

    df = nullify_strings(df)
    df = normalize_obj(df)  # trim/collapse whitespace
    # IDs/codes
    if "npi" in df:
        df["npi"] = (
            df["npi"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )
    if "zip5" in df:
        df["zip5"] = to_zip5(df["zip5"])
    if "hcpcs" in df:
        df["hcpcs"] = df["hcpcs"].astype("string").str.upper()

    # numerics
    for c in [
        "avg_submitted_charge",
        "avg_allowed_amount",
        "avg_payment_amount",
        "avg_std_amount",
    ]:
        if c in df:
            df[c] = to_float(df[c])
    for c in [
        "total_beneficiaries",
        "total_services",
        "total_bene_day_services",
        "ruca_code",
    ]:
        if c in df:
            df[c] = cast_nullable_int(df[c])

    # soft outlier trim for money columns
    for c in ["avg_submitted_charge", "avg_allowed_amount", "avg_payment_amount"]:
        if c in df and df[c].notna().any():
            df[c] = winsorize(df[c])

    # payment sanity (flag)
    violations = 0
    if {"avg_payment_amount", "avg_allowed_amount", "avg_submitted_charge"} <= set(
        df.columns
    ):
        m = (df["avg_payment_amount"] > df["avg_allowed_amount"]) | (
            df["avg_allowed_amount"] > df["avg_submitted_charge"]
        )
        violations = int(m.sum())

    # dedupe by business key (keep highest volume)
    bk = [c for c in ["npi", "hcpcs", "pos", "zip5"] if c in df.columns]
    if bk:
        sort_cols = ["total_services"] if "total_services" in df.columns else bk
        df = df.sort_values(by=sort_cols, ascending=False).drop_duplicates(
            subset=bk, keep="first"
        )

    log_run(
        LOG,
        {
            "dataset": "claims",
            "rows_out": int(len(df)),
            "payment_order_violations": violations,
        },
    )
    return df


def run():
    parts = []
    for chunk in pd.read_csv(RAW, chunksize=500_000, dtype=str, low_memory=False):
        parts.append(clean_chunk(chunk))
    out = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    log_run(
        LOG, {"dataset": "claims", "status": "SUCCESS", "total_rows": int(len(out))}
    )


if __name__ == "__main__":
    run()

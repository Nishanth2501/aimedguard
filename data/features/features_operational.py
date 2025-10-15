# data/features/features_operational.py
from __future__ import annotations
import time, json
from pathlib import Path
import numpy as np
import pandas as pd

HGI = Path("data/processed/operational/ops_hospital_info.parquet")
STATE = Path("data/processed/operational/ops_state_complications.parquet")
TEC = Path("data/processed/operational/ops_timely_effective.parquet")
OUT = Path("data/features/operational_features.parquet")
LOG = Path(f"logs/features_operational_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def run():
    hosp = pd.read_parquet(HGI)
    state = pd.read_parquet(STATE)
    tec = pd.read_parquet(TEC)

    # Ensure keys
    if "facility_id" not in hosp.columns:
        raise ValueError("ops_hospital_info.parquet must include 'facility_id'")
    if "facility_id" not in tec.columns:
        raise ValueError("ops_timely_effective.parquet must include 'facility_id'")

    # Clean numeric fields from TEC
    score_col = "score_num" if "score_num" in tec.columns else "Score"
    tec["score_num"] = pd.to_numeric(tec.get(score_col), errors="coerce")
    tec["sample_num"] = pd.to_numeric(tec.get("sample_num"), errors="coerce")

    # Aggregate timeliness per facility
    tec_agg = (
        tec.groupby("facility_id")
        .agg(
            avg_timeliness=("score_num", "mean"),
            n_measures=("measure_id", "nunique"),
            n_obs=("score_num", "count"),
        )
        .reset_index()
    )

    # Hospital rating numeric if present
    if "Hospital overall rating" in hosp.columns:
        hosp["rating_num"] = pd.to_numeric(
            hosp["Hospital overall rating"], errors="coerce"
        )
    else:
        hosp["rating_num"] = np.nan

    # Standardize state column to 'state'
    hosp["state"] = hosp["State"] if "State" in hosp.columns else np.nan

    # Merge
    df = hosp.merge(tec_agg, on="facility_id", how="left")

    # Utilization-like index (simple proxy)
    # Convert to float to avoid NA ambiguity issues
    rating_num = df["rating_num"].astype(float)
    df["utilization_index"] = np.where(
        rating_num > 0, df["avg_timeliness"] / rating_num, np.nan
    )

    # State-level complication context (pivot to a compact signal)
    if {"state", "measure_id", "n_worse"}.issubset(state.columns):
        worst = state.pivot_table(
            index="state", columns="measure_id", values="n_worse", aggfunc="mean"
        )
        worst = worst.add_prefix("state_worse_").reset_index()
        df = df.merge(worst, on="state", how="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "dataset": "operational_features",
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                }
            )
            + "\n"
        )
    print(f"✅ operational_features saved → {OUT}  rows={len(df)}")


if __name__ == "__main__":
    run()

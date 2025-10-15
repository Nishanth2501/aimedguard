# data/etl/etl_operational.py
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ._common import (
    normalize_obj,
    nullify_strings,
    cast_nullable_int,
    to_float,
    parse_date,
    period_key,
    to_zip5,
    log_run,
)

LOG_TS = time.strftime("%Y%m%d_%H%M%S")

# ---------- Complications & Deaths (State) ----------
RAW_STATE = Path("data/raw/operational/Complications_and_Deaths-State.csv")
OUT_STATE = Path("data/processed/operational/ops_state_complications.parquet")
LOG_STATE = Path(f"logs/ops_state_complications_{LOG_TS}.jsonl")


def clean_state():
    df = pd.read_csv(RAW_STATE, dtype=str)
    df = nullify_strings(df)
    df = normalize_obj(df)  # trim/case
    # counts → Int64
    for c in [
        "Number of Hospitals Worse",
        "Number of Hospitals Same",
        "Number of Hospitals Better",
        "Number of Hospitals Too Few",
    ]:
        if c in df.columns:
            df[c] = cast_nullable_int(df[c])
    # dates
    if "Start Date" in df and "End Date" in df:
        df["start_date"] = parse_date(df["Start Date"])
        df["end_date"] = parse_date(df["End Date"])
        df["period_key"] = period_key(df["start_date"])
    # rename
    df = df.rename(
        columns={
            "State": "state",
            "Measure ID": "measure_id",
            "Measure Name": "measure_name",
            "Number of Hospitals Worse": "n_worse",
            "Number of Hospitals Same": "n_same",
            "Number of Hospitals Better": "n_better",
            "Number of Hospitals Too Few": "n_too_few",
            "Footnote": "footnote",
        }
    )
    # uniqueness check
    dupe_keys = df.duplicated(
        subset=["state", "measure_id", "period_key"], keep=False
    ).sum()
    log_run(
        LOG_STATE,
        {
            "dataset": "ops_state",
            "rows_out": int(len(df)),
            "dupe_triplets": int(dupe_keys),
        },
    )
    OUT_STATE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_STATE, index=False)


# ---------- Timely & Effective Care (Hospital) ----------
RAW_TEC = Path("data/raw/operational/Timely_and_Effective_Care-Hospital.csv")
OUT_TEC = Path("data/processed/operational/ops_timely_effective.parquet")
LOG_TEC = Path(f"logs/ops_timely_effective_{LOG_TS}.jsonl")


def clean_tec():
    df = pd.read_csv(RAW_TEC, dtype=str, low_memory=False)
    df = nullify_strings(df)
    df = normalize_obj(df)
    df = df.rename(
        columns={
            "Facility ID": "facility_id",
            "Facility Name": "facility_name",
            "City/Town": "city",
            "State": "state",
            "ZIP Code": "zip5",
            "County/Parish": "county",
            "Telephone Number": "phone",
            "Measure ID": "measure_id",
            "Measure Name": "measure_name",
            "Start Date": "start_date_raw",
            "End Date": "end_date_raw",
        }
    )
    df["facility_id"] = df["facility_id"].astype("string")
    df["zip5"] = to_zip5(df["zip5"])
    df["score_num"] = pd.to_numeric(df.get("Score"), errors="coerce")
    df["sample_num"] = cast_nullable_int(df.get("Sample"))
    df["start_date"] = parse_date(df["start_date_raw"])
    df["end_date"] = parse_date(df["end_date_raw"])
    df["period_key"] = period_key(df["start_date"])
    # save
    OUT_TEC.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_TEC, index=False)
    pct_na_score = float(df["score_num"].isna().mean() * 100)
    log_run(
        LOG_TEC,
        {
            "dataset": "ops_tec",
            "rows_out": int(len(df)),
            "pct_score_na": round(pct_na_score, 2),
        },
    )


# ---------- Hospital General Info ----------
RAW_HGI = Path("data/raw/operational/Hospital_General_Information.csv")
OUT_HGI = Path("data/processed/operational/ops_hospital_info.parquet")
LOG_HGI = Path(f"logs/ops_hospital_info_{LOG_TS}.jsonl")

BOOL_MAP = {"YES": True, "Y": True, "NO": False, "N": False}


def clean_hgi():
    df = pd.read_csv(RAW_HGI, dtype=str)
    df = nullify_strings(df)
    df = normalize_obj(df)
    df = df.rename(
        columns={
            "Facility ID": "facility_id",
            "Facility Name": "facility_name",
            "City/Town": "city",
            "ZIP Code": "zip5",
            "Hospital Type": "hospital_type",
            "Hospital Ownership": "hospital_ownership",
            "Emergency Services": "emergency_services",
            "Meets criteria for birthing friendly designation": "birthing_friendly",
        }
    )
    df["facility_id"] = df["facility_id"].astype("string")
    df["zip5"] = to_zip5(df["zip5"])
    df["emergency_services"] = df["emergency_services"].map(BOOL_MAP).astype("boolean")
    df["birthing_friendly"] = df["birthing_friendly"].map(BOOL_MAP).astype("boolean")
    # convert footnote-like numeric columns to Int64 when possible
    for c in df.columns:
        if c.endswith("Footnote"):
            df[c] = cast_nullable_int(df[c])
    OUT_HGI.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_HGI, index=False)
    log_run(LOG_HGI, {"dataset": "ops_hgi", "rows_out": int(len(df))})


def run():
    clean_state()
    clean_tec()
    clean_hgi()


if __name__ == "__main__":
    run()

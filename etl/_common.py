# data/etl/_common.py
from __future__ import annotations
from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd

NULL_LIKE = {
    "",
    "NA",
    "N/A",
    "NaN",
    "nan",
    "NONE",
    "None",
    "NOT AVAILABLE",
    "Not Available",
    None,
}


def normalize_obj(
    df: pd.DataFrame, cols: list[str] | None = None, upper: bool = False
) -> pd.DataFrame:
    cols = cols or list(df.select_dtypes(include="object").columns)
    for c in cols:
        s = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        df[c] = s.str.upper() if upper else s
    return df


def to_zip5(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(r"\D", "", regex=True)
    return s.str.zfill(5).where(s.notna())


def cast_nullable_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def winsorize(s: pd.Series, lo=0.005, hi=0.995) -> pd.Series:
    x = s.copy()
    ql, qh = x.quantile(lo), x.quantile(hi)
    return x.clip(lower=ql, upper=qh)


def parse_date(s: pd.Series, dayfirst=False) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def period_key(start: pd.Series) -> pd.Series:
    d = parse_date(start)
    return d.dt.to_period("Q").astype(str)


def nullify_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].replace(list(NULL_LIKE), np.nan)
    return df


def log_run(log_path: Path | str, payload: dict) -> None:
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")

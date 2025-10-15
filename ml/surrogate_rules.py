# ml/surrogate_rules.py
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

DATA = Path("data/features/final_feature_mart.parquet")
FRAUD_MODEL = Path("models/fraud_baseline.joblib")
OPS_MODEL = Path("models/ops_forecast.joblib")
OUTDIR = Path("models/explain/rules")

MAX_DEPTH = 3
SAMPLE_N = 10000  # keep it quick


def _prep_X(df: pd.DataFrame, drop_cols: set[str]) -> pd.DataFrame:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return X.select_dtypes(include=[np.number]).fillna(0.0)


def _save_rules_txt(
    tree: DecisionTreeRegressor, feature_names: list[str], out_path: Path
) -> None:
    rules = export_text(tree, feature_names=feature_names)
    out_path.write_text(rules, encoding="utf-8")
    print(f"🧾 rules saved → {out_path}")


def run():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing feature mart: {DATA}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)

    # ---------- FRAUD ----------
    if FRAUD_MODEL.exists():
        print("Building surrogate rules for FRAUD model")
        m = joblib.load(FRAUD_MODEL)

        df_f = df.copy()
        df_f["fraud_flag"] = (
            (df_f["pay_ratio"] > 1.1) & (df_f["svc_per_bene"] > 10)
        ).astype(int)
        drop_cols = {"npi", "dom_state", "fraud_flag"}
        X = _prep_X(df_f, drop_cols)

        # use model outputs as target (probability preferred)
        y_hat = (
            m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X)
        )

        n = min(SAMPLE_N, len(X))
        Xs = X.sample(n=n, random_state=42)
        y_s = y_hat[Xs.index]

        tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=42)
        tree.fit(Xs, y_s)
        _save_rules_txt(tree, list(Xs.columns), OUTDIR / "fraud_surrogate_rules.txt")
    else:
        print("Info: fraud model not found; skipping.")

    # ---------- OPS ----------
    if OPS_MODEL.exists():
        print("Building surrogate rules for OPS model")
        m = joblib.load(OPS_MODEL)
        target = (
            "utilization_index" if "utilization_index" in df.columns else "pay_ratio"
        )
        drop_cols = {"npi", "dom_state", target}
        X = _prep_X(df, drop_cols)
        y_hat = m.predict(X)

        n = min(SAMPLE_N, len(X))
        Xs = X.sample(n=n, random_state=42)
        y_s = y_hat[Xs.index]

        tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=42)
        tree.fit(Xs, y_s)
        _save_rules_txt(tree, list(Xs.columns), OUTDIR / "ops_surrogate_rules.txt")
    else:
        print("Info: ops model not found; skipping.")

    print("Success: surrogate rule extraction complete")


if __name__ == "__main__":
    run()

# ml/explain_global_fast.py
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

DATA = Path("data/features/final_feature_mart.parquet")
FRAUD_MODEL = Path("models/fraud_baseline.joblib")
OPS_MODEL = Path("models/ops_forecast.joblib")
OUTDIR = Path("models/explain/fast")

SAMPLE_N = 5000  # keep it fast and consistent
N_REPEATS = 5  # permutation repeats (good balance of stability vs speed)
GRID_RES = 20  # PDP grid resolution


def _prep_X(df: pd.DataFrame, drop_cols: set[str]) -> pd.DataFrame:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return X.select_dtypes(include=[np.number]).fillna(0.0)


def _perm_importance(
    estimator, X: pd.DataFrame, y: pd.Series | np.ndarray, kind: str
) -> pd.DataFrame:
    """Model-agnostic global importance via permutation."""
    result = permutation_importance(
        estimator, X, y, n_repeats=N_REPEATS, random_state=42, n_jobs=1
    )
    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "import_mean": result.importances_mean,
            "import_std": result.importances_std,
        }
    ).sort_values("import_mean", ascending=False)
    out_csv = OUTDIR / f"{kind}_perm_importance.csv"
    imp.to_csv(out_csv, index=False)
    print(f"Saved permutation importance to {out_csv}")
    return imp


def _pdp_top3(estimator, X: pd.DataFrame, top_features: list[str], kind: str) -> None:
    feats = list(top_features[:3])
    if not feats:
        print("Info: no features to plot for PDP")
        return
    plt.figure(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        estimator, X, feats, kind="average", grid_resolution=GRID_RES
    )
    out_png = OUTDIR / f"{kind}_pdp_top3.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Saved PDP (top 3) to {out_png}")


def run():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing feature mart: {DATA}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)

    # ---------- FRAUD (classification) ----------
    if FRAUD_MODEL.exists():
        print("Processing fraud model — permutation importance + PDP")
        model_f = joblib.load(FRAUD_MODEL)

        # Recreate synthetic target exactly like training script
        df_f = df.copy()
        df_f["fraud_flag"] = (
            (df_f["pay_ratio"] > 1.1) & (df_f["svc_per_bene"] > 10)
        ).astype(int)

        drop_cols_f = {"npi", "dom_state", "fraud_flag"}
        Xf = _prep_X(df_f, drop_cols_f)
        yf = df_f["fraud_flag"].astype(int)

        # sample for speed
        n = min(SAMPLE_N, len(Xf))
        Xf_s = Xf.sample(n=n, random_state=42)
        yf_s = yf.loc[Xf_s.index]

        imp_f = _perm_importance(model_f, Xf_s, yf_s, kind="fraud")
        _pdp_top3(model_f, Xf_s, imp_f["feature"].tolist(), kind="fraud")
    else:
        print("Info: fraud model not found; skipping fraud explainability.")

    # ---------- OPS (regression) ----------
    if OPS_MODEL.exists():
        print("Processing ops model — permutation importance + PDP")
        model_o = joblib.load(OPS_MODEL)
        target = (
            "utilization_index" if "utilization_index" in df.columns else "pay_ratio"
        )
        drop_cols_o = {"npi", "dom_state", target}
        Xo = _prep_X(df, drop_cols_o)
        yo = df[target].astype(float).fillna(0.0)

        n = min(SAMPLE_N, len(Xo))
        Xo_s = Xo.sample(n=n, random_state=42)
        yo_s = yo.loc[Xo_s.index]

        imp_o = _perm_importance(model_o, Xo_s, yo_s, kind="ops")
        _pdp_top3(model_o, Xo_s, imp_o["feature"].tolist(), kind="ops")
    else:
        print("Info: ops model not found; skipping ops explainability.")

    meta = {
        "rows_total": int(len(df)),
        "sample_used": SAMPLE_N,
        "perm_repeats": N_REPEATS,
        "pdp_grid_resolution": GRID_RES,
    }
    (OUTDIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Success: fast global explainability artifacts written")


if __name__ == "__main__":
    run()

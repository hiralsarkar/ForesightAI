"""Probability calibration and the final fitted artefact.

Why this module exists: the product does not display a ranking, it displays a **score**.
Modules 3, 4 and 5 put a 0-100 number on screen, run what-if slider changes against it,
and stress-test it. A number that only ranks correctly is not enough -- if the gauge
reads 81, roughly 81% of companies scoring 81 should actually be in distress, otherwise
every downstream narrative ("score moves 81 -> 91 under rate stress") is theatre.

`scale_pos_weight` buys us little on PR-AUC (measured ~neutral) but it *does* inflate
predicted probabilities, because it tells the booster positives are ~25x more common
than they are. So we keep class weighting for the operating-point behaviour and then
correct the probabilities with `CalibratedClassifierCV`.

Calibration is fit on held-out folds (`cv=5` internally), never on the training data the
base model already saw -- otherwise the calibrator learns the model's overfit
confidence and reports it back as truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..features.polish_schema import TARGET, Scope
from .train import BUILDERS, RANDOM_STATE, build_pipeline, get_features

Method = Literal["isotonic", "sigmoid"]

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass
class CalibrationReport:
    """Before/after calibration quality. Brier is the one that should improve."""

    method: str
    brier_raw: float
    brier_calibrated: float
    pr_auc_raw: float
    pr_auc_calibrated: float
    roc_auc_raw: float
    roc_auc_calibrated: float
    n_holdout: int

    @property
    def brier_improvement(self) -> float:
        return self.brier_raw - self.brier_calibrated

    def summary(self) -> dict[str, Any]:
        d = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}
        d["brier_improvement"] = round(self.brier_improvement, 5)
        return d

    def __str__(self) -> str:
        return (
            f"calibration[{self.method}]  "
            f"Brier {self.brier_raw:.5f} -> {self.brier_calibrated:.5f} "
            f"({self.brier_improvement:+.5f})   "
            f"PR-AUC {self.pr_auc_raw:.4f} -> {self.pr_auc_calibrated:.4f}"
        )


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Predicted vs observed distress rate per probability bin.

    This is the evidence that the gauge is calibrated. `gap` near zero across bins means a
    displayed score of 80 corresponds to a real 80% distress rate. Quantile bins are
    used rather than uniform ones because at a 3.9% base rate almost every prediction
    lands in the lowest uniform bin.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], bins=n_bins)

    out = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    out["gap"] = (out["predicted"] - out["observed"]).round(4)
    out["predicted"] = out["predicted"].round(4)
    out["observed"] = out["observed"].round(4)
    return out


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Weighted mean |predicted - observed| across bins. Lower is better."""
    tbl = reliability_table(y_true, y_prob, n_bins)
    if tbl.empty:
        return float("nan")
    weights = tbl["n"] / tbl["n"].sum()
    return float((weights * (tbl["predicted"] - tbl["observed"]).abs()).sum())


def fit_calibrated(
    df: pd.DataFrame,
    model_name: str = "lightgbm",
    feature_set: str = "serving",
    method: Method = "sigmoid",
    use_smote: bool = False,
    holdout_size: float = 0.2,
    target: str = TARGET,
    monotone: Scope = "slider",
    **model_overrides,
) -> tuple[Any, CalibrationReport, list[str]]:
    """Fit a calibrated distress model and report what calibration bought.

    Returns `(fitted_estimator, report, feature_names)`. The estimator exposes
    `predict_proba` and is what the dashboard should load -- never the raw booster.

    A stratified holdout is carved out first so the report describes data neither the
    base model nor the calibrator has seen.
    """
    features = get_features(df, feature_set)
    X = df[features].to_numpy(dtype="float64")
    y = df[target].to_numpy(dtype="int8")

    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=holdout_size, stratify=y, random_state=RANDOM_STATE
    )

    def _pipe():
        return build_pipeline(
            BUILDERS[model_name](
                y_fit, features=features, scope=monotone, **model_overrides
            ),
            use_smote=use_smote,
        )

    raw = _pipe().fit(X_fit, y_fit)
    p_raw = raw.predict_proba(X_hold)[:, 1]

    # cv=5 -> the calibrator is trained on out-of-fold predictions of clones of the
    # pipeline, so it never calibrates on data the base estimator memorised.
    calibrated = CalibratedClassifierCV(_pipe(), method=method, cv=5)
    calibrated.fit(X_fit, y_fit)
    p_cal = calibrated.predict_proba(X_hold)[:, 1]

    report = CalibrationReport(
        method=method,
        brier_raw=float(brier_score_loss(y_hold, p_raw)),
        brier_calibrated=float(brier_score_loss(y_hold, p_cal)),
        pr_auc_raw=float(average_precision_score(y_hold, p_raw)),
        pr_auc_calibrated=float(average_precision_score(y_hold, p_cal)),
        roc_auc_raw=float(roc_auc_score(y_hold, p_raw)),
        roc_auc_calibrated=float(roc_auc_score(y_hold, p_cal)),
        n_holdout=int(len(y_hold)),
    )
    return calibrated, report, features


def compare_methods(
    df: pd.DataFrame,
    model_name: str = "lightgbm",
    feature_set: str = "serving",
    **kw,
) -> pd.DataFrame:
    """Isotonic vs sigmoid vs uncalibrated, so the choice is evidenced not assumed."""
    rows = []
    for method in ("isotonic", "sigmoid"):
        _, rep, _ = fit_calibrated(
            df, model_name=model_name, feature_set=feature_set, method=method, **kw
        )
        rows.append(rep.summary())
    return pd.DataFrame(rows)


def risk_score(probability: float | np.ndarray) -> float | np.ndarray:
    """Convert a calibrated distress probability to the displayed 0-100 risk score.

    Deliberately a straight linear mapping. Any curve applied here would break the
    interpretation the whole product rests on -- that a score of 81 means an 81%
    modelled probability of distress. If the gauge needs different visual spacing, do
    that in the gauge, not in the number.
    """
    return np.clip(np.asarray(probability, dtype="float64") * 100.0, 0.0, 100.0)


#: Score bands shown on the gauge, in the language the dashboard uses.
BANDS: tuple[tuple[float, str], ...] = (
    (25.0, "Healthy"),
    (50.0, "Watch"),
    (75.0, "Elevated Risk"),
    (100.1, "Critical"),
)


def band(score: float) -> str:
    for upper, name in BANDS:
        if score < upper:
            return name
    return BANDS[-1][1]


def save(estimator, features: list[str], report: CalibrationReport, name: str = "distress_model") -> Path:
    """Persist the fitted model plus the metadata the dashboard needs to use it safely."""
    import joblib

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump({"estimator": estimator, "features": features}, path)
    (MODEL_DIR / f"{name}_calibration.json").write_text(
        json.dumps(report.summary(), indent=2), encoding="utf-8"
    )
    return path

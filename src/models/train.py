"""Distress model training and evaluation.

the training protocol as **amended by measurement** (see docs/phase1_findings.md
section 6): evaluated on AUC-ROC and Precision-Recall, never accuracy, on stratified
5-fold CV -- but **class weighting only, SMOTE off by default**.

The blueprint mandated SMOTE + class weighting. A full 2x2 ablation across two model
families and three horizons showed SMOTE costs 12-14 PR-AUC points *even with class
weighting disabled*, so this is not merely a double-correction artefact. SMOTE
interpolates synthetic minority points in a 63-dimensional ratio space with extreme
outliers and up to 39% missingness, producing balance sheets that do not exist.
`use_smote` remains a parameter so the ablation stays reproducible on demand.

Two details are enforced
in code rather than left to discipline:

1. **SMOTE runs inside the CV fold, never before the split.** Oversampling before
   splitting leaks synthetic neighbours of validation rows into training and inflates
   every metric. We use an imblearn Pipeline so the resampler is fit on the training
   fold only.

2. **SMOTE is never applied to the validation fold.** imblearn Pipelines skip
   resamplers at transform/predict time by design, so validation keeps the true 3.9%
   distress rate and the reported PR-AUC is real.

The headline metric is **average precision (PR-AUC)**, because at a 3.9% base rate
ROC-AUC flatters a model badly. Both are reported; PR-AUC is the one that decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from ..features.polish_schema import (
    TARGET,
    Scope,
    monotone_vector,
    serving_feature_set,
)

RANDOM_STATE = 42
N_SPLITS = 5


@dataclass
class FoldResult:
    fold: int
    roc_auc: float
    pr_auc: float
    brier: float
    n_train: int
    n_valid: int
    n_distress_valid: int


@dataclass
class CVResult:
    """Cross-validation outcome. `pr_auc` is the headline; report it first."""

    model_name: str
    feature_set: str
    n_features: int
    folds: list[FoldResult] = field(default_factory=list)
    oof_pred: np.ndarray | None = None
    oof_true: np.ndarray | None = None

    @property
    def pr_auc(self) -> float:
        return float(np.mean([f.pr_auc for f in self.folds]))

    @property
    def pr_auc_std(self) -> float:
        return float(np.std([f.pr_auc for f in self.folds]))

    @property
    def roc_auc(self) -> float:
        return float(np.mean([f.roc_auc for f in self.folds]))

    @property
    def roc_auc_std(self) -> float:
        return float(np.std([f.roc_auc for f in self.folds]))

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "feature_set": self.feature_set,
            "n_features": self.n_features,
            "pr_auc": round(self.pr_auc, 4),
            "pr_auc_std": round(self.pr_auc_std, 4),
            "roc_auc": round(self.roc_auc, 4),
            "roc_auc_std": round(self.roc_auc_std, 4),
            "brier": round(float(np.mean([f.brier for f in self.folds])), 5),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"{s['model']:<10} [{s['feature_set']}, {s['n_features']}f]  "
            f"PR-AUC {s['pr_auc']:.4f} +/-{s['pr_auc_std']:.4f}   "
            f"ROC-AUC {s['roc_auc']:.4f} +/-{s['roc_auc_std']:.4f}"
        )


def get_features(df: pd.DataFrame, feature_set: str = "serving") -> list[str]:
    """Resolve a named feature set to column names.

    `serving` is the default and the safe one: it excludes attributes we cannot
    rebuild for Indian companies, so the trained model can actually score the demo
. `all` trains on every attribute and is for benchmarking only -- a model
    trained on `all` must never be shipped to the dashboard.
    """
    attrs = [c for c in df.columns if c.startswith("Attr")]
    if feature_set == "all":
        return attrs
    if feature_set == "serving":
        allowed = set(serving_feature_set(include_derived=True))
        return [c for c in attrs if c in allowed]
    if feature_set == "serving_direct":
        allowed = set(serving_feature_set(include_derived=False))
        return [c for c in attrs if c in allowed]
    if feature_set == "screener":
        # The features the Screener serving bridge can actually populate. Training on
        # this set is the real fix for train/serve parity: the model can only
        # depend on features that exist when scoring an Indian company. Imported lazily
        # to keep the models layer independent of the serving layer.
        from ..serving.screener import screener_feature_set

        allowed = set(screener_feature_set())
        return [c for c in attrs if c in allowed]
    raise ValueError(f"unknown feature_set {feature_set!r}")


def scale_pos_weight(y: np.ndarray) -> float:
    """Class-weight term for gradient boosters: negatives / positives."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / pos if pos else 1.0


def build_pipeline(
    estimator,
    use_smote: bool = False,
    smote_k: int = 5,
    impute: bool | None = None,
) -> ImbPipeline:
    """(Impute) -> (SMOTE) -> estimator, as a leak-safe imblearn Pipeline.

    **Imputation is off by default and that is a deliberate, measured choice.**
    Missingness in this dataset is informative, not random: distressed companies fail
    to report. `Attr27` (interest coverage) is absent for 311 firms, and its absence
    carries signal that median-imputation erases. Letting XGBoost and LightGBM route
    NaN natively beat median imputation in all six tested cells (+0.014 to +0.031
    PR-AUC across two model families and three horizons).

    It also fixes a scenario-analysis bug: with imputation, setting interest coverage
    to a *worse* value could *lower* the displayed risk score, because the imputed
    median was more benign than "missing". Native NaN handling makes the what-if
    calculator behave the way a slider must.

    SMOTE cannot consume NaN, so imputation is force-enabled when SMOTE is on. Pass
    `impute=True` explicitly to override.
    """
    if impute is None:
        impute = use_smote  # only impute when the resampler requires it

    steps: list[tuple[str, Any]] = []
    if impute:
        # Fit inside the fold; fitting before the split would leak the validation
        # distribution into training.
        steps.append(("impute", SimpleImputer(strategy="median")))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def make_xgb(
    y: np.ndarray,
    features: Sequence[str] | None = None,
    scope: Scope = "slider",
    **overrides,
):
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=3,
        # Class weighting, per core requirement as amended (SMOTE off).
        scale_pos_weight=scale_pos_weight(y),
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if features is not None:
        # XGBoost wants a parenthesised string, positionally aligned to the matrix.
        params["monotone_constraints"] = "(" + ",".join(
            str(v) for v in monotone_vector(features, scope)
        ) + ")"
    params.update(overrides)
    return XGBClassifier(**params)


def make_lgbm(
    y: np.ndarray,
    features: Sequence[str] | None = None,
    scope: Scope = "slider",
    **overrides,
):
    from lightgbm import LGBMClassifier

    params = dict(
        n_estimators=400,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight(y),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    if features is not None:
        params["monotone_constraints"] = list(monotone_vector(features, scope))
        # 'basic' enforces monotonicity by clamping split gains and is noticeably
        # lossy; 'advanced' preserves far more accuracy for the same guarantee.
        params["monotone_constraints_method"] = "advanced"
    params.update(overrides)
    return LGBMClassifier(**params)


BUILDERS = {"xgboost": make_xgb, "lightgbm": make_lgbm}


def cross_validate(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    feature_set: str = "serving",
    use_smote: bool = False,
    n_splits: int = N_SPLITS,
    target: str = TARGET,
    monotone: Scope = "slider",
    **model_overrides,
) -> CVResult:
    """Stratified k-fold CV with fold-internal imputation and resampling.

    `monotone` scopes the economic-direction constraint: `"slider"` (default) covers
    user-facing metrics, `"all"` every directional attribute, `"none"` disables it.
    Measured cost at 1year/lightgbm: none 0.803, slider 0.780, all 0.702.
    """
    if model_name not in BUILDERS:
        raise ValueError(f"model_name must be one of {sorted(BUILDERS)}")

    features = get_features(df, feature_set)
    if not features:
        raise ValueError(f"feature set {feature_set!r} resolved to no columns")

    X = df[features].to_numpy(dtype="float64")
    y = df[target].to_numpy(dtype="int8")

    result = CVResult(model_name=model_name, feature_set=feature_set, n_features=len(features))
    oof = np.full(len(y), np.nan, dtype="float64")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for i, (tr, va) in enumerate(cv.split(X, y), start=1):
        # scale_pos_weight is computed on the *training fold* only. `features` is
        # passed so the constraint vector aligns with the matrix column order.
        estimator = BUILDERS[model_name](
            y[tr], features=features, scope=monotone, **model_overrides
        )
        pipe = build_pipeline(estimator, use_smote=use_smote)
        pipe.fit(X[tr], y[tr])

        # Validation fold keeps its true class balance -- no resampling at predict time.
        p = pipe.predict_proba(X[va])[:, 1]
        oof[va] = p

        result.folds.append(
            FoldResult(
                fold=i,
                roc_auc=float(roc_auc_score(y[va], p)),
                pr_auc=float(average_precision_score(y[va], p)),
                brier=float(brier_score_loss(y[va], p)),
                n_train=int(len(tr)),
                n_valid=int(len(va)),
                n_distress_valid=int((y[va] == 1).sum()),
            )
        )

    result.oof_pred = oof
    result.oof_true = y
    return result


def operating_points(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Precision/recall at candidate thresholds, for choosing a credit-policy cutoff.

    A credit team does not consume a probability; it consumes a review queue. This
    table answers "if we review the top N%, how many real distress cases do we catch,
    and how many false alarms do we absorb?" -- the form the Module 8 portfolio view
    and the demo narrative both need.
    """
    precision, recall, thresh = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]

    rows = []
    for target_recall in (0.50, 0.60, 0.70, 0.80, 0.90):
        idx = np.where(recall >= target_recall)[0]
        if len(idx) == 0:
            continue
        best = idx[np.argmax(precision[idx])]
        cutoff = float(thresh[best])
        flagged = int((y_score >= cutoff).sum())
        rows.append(
            {
                "target_recall": target_recall,
                "threshold": round(cutoff, 4),
                "precision": round(float(precision[best]), 4),
                "recall": round(float(recall[best]), 4),
                "flagged": flagged,
                "flagged_pct": round(100 * flagged / len(y_score), 2),
            }
        )
    return pd.DataFrame(rows)


def review_budget_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    budgets: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Precision and catch-rate at fixed **review budgets** (top-K% by risk score).

    `operating_points` anchors on recall ("to catch 60%, what queue?"). This anchors on
    *capacity* -- the way a risk officer actually plans: "we can review K companies this
    cycle; rank by risk, take the top K, and tell me how many real distress cases that
    queue catches and how many reviews are wasted." Score-agnostic, so it ranks our model
    or an Altman -Z screen identically for a like-for-like comparison at the same budget.
    """
    y_true = np.asarray(y_true, dtype="int8")
    y_score = np.asarray(y_score, dtype="float64")
    n = len(y_score)
    total_pos = int(y_true.sum())
    base_rate = total_pos / n if n else float("nan")
    order = np.argsort(-y_score, kind="stable")  # highest risk first

    rows = []
    for b in budgets:
        k = max(1, int(round(b * n)))
        top = order[:k]
        caught = int(y_true[top].sum())
        precision = caught / k
        rows.append(
            {
                "budget_pct": round(100 * b, 1),
                "reviewed": k,
                "caught": caught,
                "missed": total_pos - caught,
                "precision": round(precision, 4),
                "catch_rate": round(caught / total_pos, 4) if total_pos else float("nan"),
                "wasted_reviews": k - caught,
                "lift_vs_random": round(precision / base_rate, 2) if base_rate else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def expected_cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_per_miss: float,
    cost_per_review: float,
    n_points: int = 201,
) -> pd.DataFrame:
    """Expected review-policy cost across every budget, to pick the cost-minimising queue.

    Turns the operating point into a decision. A risk officer faces a real tradeoff: every
    company reviewed costs `cost_per_review` (analyst time), and every distress case left
    un-reviewed costs `cost_per_miss` (the exposure on a default nobody caught). Ranking by
    risk and reviewing the top K, the expected cost is

        cost(K) = cost_per_miss * (distress still in the un-reviewed tail) + cost_per_review * K

    The K that minimises this is the recommended review budget. `caught` uses a cumulative
    sum over the risk-sorted labels, so the whole curve is one pass.
    """
    y_true = np.asarray(y_true, dtype="int8")
    y_score = np.asarray(y_score, dtype="float64")
    n = len(y_score)
    total_pos = int(y_true.sum())
    order = np.argsort(-y_score, kind="stable")  # highest risk first
    caught_cum = np.concatenate([[0], np.cumsum(y_true[order])])  # caught after reviewing k

    ks = np.unique(np.clip(np.round(np.linspace(0, n, n_points)).astype(int), 0, n))
    rows = []
    for k in ks:
        caught = int(caught_cum[k])
        missed = total_pos - caught
        cost = cost_per_miss * missed + cost_per_review * int(k)
        rows.append(
            {
                "reviewed": int(k),
                "budget_pct": round(100 * k / n, 2),
                "caught": caught,
                "missed": missed,
                "cost": float(cost),
            }
        )
    out = pd.DataFrame(rows)
    out["optimal"] = out["cost"] == out["cost"].min()
    return out


def baseline_metrics(y: np.ndarray) -> dict[str, float]:
    """Reference points every reported metric should be read against."""
    n = len(y)
    pos = int((y == 1).sum())
    return {
        "n": n,
        "distress": pos,
        "base_rate": round(pos / n, 5),
        # A random model's PR-AUC equals the base rate; ROC-AUC equals 0.5.
        "random_pr_auc": round(pos / n, 5),
        "random_roc_auc": 0.5,
        # The number that shows why accuracy is banned.
        "majority_class_accuracy": round((n - pos) / n, 5),
    }

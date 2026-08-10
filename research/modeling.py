"""Distress model training and evaluation.

The training protocol, **as amended by measurement** (see docs/model_evaluation.md
section 6): evaluated on AUC-ROC and Precision-Recall, never accuracy, on stratified
5-fold CV -- but **class weighting only, SMOTE off by default**.

SMOTE plus class weighting is the obvious first move. A full 2x2 ablation across two model
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


from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from foresight import (serving_feature_set, monotone_vector, Category, BY_ID,
                       label_for, rename_for_display, Serving, Attribute, Scope,
                       TARGET, direction_for, missingness)


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
        # Class weighting, as amended by the ablation (SMOTE off).
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
    and how many false alarms do we absorb?" -- the form the portfolio view
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


import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


Method = Literal["isotonic", "sigmoid"]

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


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


def save_calibrated(estimator, features: list[str], report: CalibrationReport, name: str = "distress_model") -> Path:
    """Persist the fitted model plus the metadata the dashboard needs to use it safely."""
    import joblib

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump({"estimator": estimator, "features": features}, path)
    (MODEL_DIR / f"{name}_calibration.json").write_text(
        json.dumps(report.summary(), indent=2), encoding="utf-8"
    )
    return path

"""Optuna hyperparameter search.

Methodology, because the number this produces is only worth quoting if the protocol is
clean:

**Tuning and reporting use disjoint data.** A stratified test split is carved off first
and Optuna never sees it. The search optimises 5-fold CV PR-AUC on the training portion
only; the reported improvement is measured on the held-out test set. Tuning against the
same folds you then report is a well-known optimistic bias -- the search selects
hyperparameters that suit those particular folds, and the "improvement" is partly
memorisation of the split.

**The shipping config is held fixed.** The search tunes tree/regularisation shape only.
Class weighting, native NaN handling, and the slider-scoped monotonic constraints are
decisions already made against evidence and are not
re-litigated by a random search optimising a single metric. `scale_pos_weight` and
`monotone_constraints` are therefore excluded from the search space -- the builders set
them per-fold, and letting Optuna override them would silently undo the demo-safety
guarantee.

**PR-AUC is the objective**, matching the headline metric. Optimising ROC-AUC at a 3.9%
base rate would tune for the wrong thing.
"""


import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RESULTS_DIR = Path(__file__).resolve().parents[1] / "models"

#: Never tunable -- these encode decisions made against evidence, not search.
PROTECTED_PARAMS = frozenset(
    {"scale_pos_weight", "monotone_constraints", "monotone_constraints_method", "random_state"}
)


@dataclass
class TuningResult:
    model_name: str
    n_trials: int
    best_params: dict[str, Any]
    best_cv_pr_auc: float
    baseline_test_pr_auc: float
    tuned_test_pr_auc: float
    baseline_test_roc_auc: float
    tuned_test_roc_auc: float
    n_train: int
    n_test: int

    @property
    def test_improvement(self) -> float:
        return self.tuned_test_pr_auc - self.baseline_test_pr_auc

    @property
    def helped(self) -> bool:
        return self.test_improvement > 0

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["test_improvement"] = round(self.test_improvement, 5)
        d["helped"] = self.helped
        for k in ("best_cv_pr_auc", "baseline_test_pr_auc", "tuned_test_pr_auc",
                  "baseline_test_roc_auc", "tuned_test_roc_auc"):
            d[k] = round(d[k], 5)
        return d

    def __str__(self) -> str:
        verdict = "improved" if self.helped else "did NOT improve"
        return (
            f"optuna[{self.model_name}, {self.n_trials} trials]  "
            f"held-out PR-AUC {self.baseline_test_pr_auc:.4f} -> "
            f"{self.tuned_test_pr_auc:.4f} ({self.test_improvement:+.4f}) -- {verdict}"
        )


def _lgbm_space(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 80),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        # LightGBM ignores `subsample` unless `subsample_freq` >= 1. The hand-set
        # baseline specified subsample=0.8 with no freq, so it was never actually
        # bagging -- a latent no-op the search makes explicit.
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _xgb_space(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


SPACES: dict[str, Callable] = {"lightgbm": _lgbm_space, "xgboost": _xgb_space}


def _assert_unprotected(params: dict[str, Any]) -> None:
    leaked = PROTECTED_PARAMS & set(params)
    if leaked:
        raise ValueError(
            f"search space must not tune {sorted(leaked)} -- these encode evidenced "
            "decisions (class weighting, monotonic constraints) and are set per-fold"
        )


def tune(
    df: pd.DataFrame,
    model_name: str = "lightgbm",
    feature_set: str = "serving",
    n_trials: int = 50,
    n_splits: int = 5,
    test_size: float = 0.2,
    monotone: Scope = "slider",
    target: str = TARGET,
    seed: int = RANDOM_STATE,
    show_progress: bool = False,
):
    """Run the search and evaluate the winner on data it never saw.

    Returns `(TuningResult, study)`. Requires >= 50 trials to count as a real search.
    """
    import optuna

    if n_trials < 50:
        raise ValueError(f"need >= 50 trials for a meaningful search, got {n_trials}")
    if model_name not in SPACES:
        raise ValueError(f"model_name must be one of {sorted(SPACES)}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    # Held out before anything else. The search never touches this.
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed
    )

    def objective(trial) -> float:
        params = SPACES[model_name](trial)
        _assert_unprotected(params)
        return cross_validate(
            train_df,
            model_name=model_name,
            feature_set=feature_set,
            n_splits=n_splits,
            monotone=monotone,
            target=target,
            **params,
        ).pr_auc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"foresight_{model_name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    baseline = _fit_and_score(train_df, test_df, model_name, feature_set, monotone, target, {})
    tuned = _fit_and_score(
        train_df, test_df, model_name, feature_set, monotone, target, study.best_params
    )

    result = TuningResult(
        model_name=model_name,
        n_trials=n_trials,
        best_params=study.best_params,
        best_cv_pr_auc=float(study.best_value),
        baseline_test_pr_auc=baseline["pr_auc"],
        tuned_test_pr_auc=tuned["pr_auc"],
        baseline_test_roc_auc=baseline["roc_auc"],
        tuned_test_roc_auc=tuned["roc_auc"],
        n_train=len(train_df),
        n_test=len(test_df),
    )
    return result, study


def _fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    feature_set: str,
    monotone: Scope,
    target: str,
    params: dict[str, Any],
) -> dict[str, float]:
    """Fit on train, score the untouched test split."""
    from sklearn.metrics import average_precision_score, roc_auc_score


    features = get_features(train_df, feature_set)
    X_tr = train_df[features].to_numpy("float64")
    y_tr = train_df[target].to_numpy("int8")
    X_te = test_df[features].to_numpy("float64")
    y_te = test_df[target].to_numpy("int8")

    est = BUILDERS[model_name](y_tr, features=features, scope=monotone, **params)
    pipe = build_pipeline(est, use_smote=False).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]

    return {
        "pr_auc": float(average_precision_score(y_te, p)),
        "roc_auc": float(roc_auc_score(y_te, p)),
    }


def trials_frame(study) -> pd.DataFrame:
    """All trials, best first -- the audit trail we log for every search."""
    rows = [
        {"trial": t.number, "pr_auc": t.value, "state": t.state.name, **t.params}
        for t in study.trials
        if t.value is not None
    ]
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def param_importance(study) -> pd.DataFrame:
    """Which hyperparameters actually mattered."""
    import optuna

    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        return pd.DataFrame(columns=["parameter", "importance"])
    return pd.DataFrame(
        [{"parameter": k, "importance": round(v, 4)} for k, v in imp.items()]
    )


def save_tuning(result: TuningResult, study, name: str | None = None) -> Path:
    """Persist the result and the full trial log."""
    name = name or f"optuna_{result.model_name}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(result.summary(), indent=2), encoding="utf-8"
    )
    trials_frame(study).to_csv(RESULTS_DIR / f"{name}_trials.csv", index=False)
    return RESULTS_DIR / f"{name}.json"

"""SHAP explainability with business-language labels.

The rule: no cryptic variable names on any chart the user will see. `Attr27`
means nothing; "Interest Coverage Ratio" means everything. Every public function here
returns labelled output, and `polish_schema.LABELS` is the single mapping.

Two implementation notes that matter:

* **TreeExplainer needs the raw booster, not the calibrated wrapper.**
  `CalibratedClassifierCV` wraps clones of the pipeline, so we reach through to an
  underlying tree model. `explainer_for` handles the unwrapping and raises a clear
  error rather than silently falling back to a slow model-agnostic explainer.

* **SHAP explains the ranking model, the score comes from the calibrated one.** These
  differ by a monotonic transform (the sigmoid calibrator), so contribution *directions*
  and relative magnitudes stay valid, which is what the waterfall communicates. The SHAP
  bars do not sum exactly to the number on the gauge, and the code does not claim they do.

* **We explain the whole ensemble, not one fold.** `CalibratedClassifierCV(cv=5)` fits five
  pipelines and averages them, so the ranking behaviour behind the score is the *mean* of
  five boosters. `_fold_boosters` returns all five and `shap_values` averages their SHAP.
  TreeExplainer works in log-odds (margin) space, and each booster's SHAP sums exactly to
  its own margin, so the fold-averaged SHAP sums exactly to the boosters' mean margin --
  verified to ~1e-15. That mean margin *is* the ensemble's ranking model (its sigmoid tracks
  the averaged probability to <1e-2), so the waterfall now decomposes the model behind the
  score, not one fold that approximates it. (This closes the earlier "one fold's booster"
  caveat; the only remaining, inherent gap to the displayed 0-100 number is the monotonic
  calibration transform noted above.)

The three-sentence summary is deliberately **rule-based, not LLM-generated**. It appears
next to a number a credit committee may act on, so it must be reproducible and incapable
of inventing a fact.
"""


from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class Contribution:
    """One feature's push on one prediction."""

    feature: str  # raw id, e.g. "Attr27"
    label: str  # business English, e.g. "Interest Coverage Ratio"
    value: float  # the company's actual value
    shap: float  # signed contribution; positive = pushes toward distress
    category: str

    @property
    def direction(self) -> str:
        return "increases risk" if self.shap > 0 else "reduces risk"


def _unwrap_pipeline(estimator: Any) -> Any:
    """Reach the tree booster inside a (possibly pipeline-wrapped) estimator."""
    seen = estimator
    if hasattr(seen, "named_steps"):
        seen = seen.named_steps.get("model", seen)
    if not hasattr(seen, "predict_proba"):
        raise TypeError(f"could not locate a fitted tree model inside {type(estimator).__name__}")
    return seen


def _fold_boosters(estimator: Any) -> list[Any]:
    """Every tree booster behind the scoring ensemble.

    `CalibratedClassifierCV(cv=5)` averages five fitted pipelines; we return the booster
    from each so an explanation can average over the same five, faithfully describing the
    ranking model the ensemble applies rather than one fold that only approximates it. A
    plain pipeline (or bare booster) yields a single-element list.
    """
    if hasattr(estimator, "calibrated_classifiers_") and estimator.calibrated_classifiers_:
        boosters = []
        for cc in estimator.calibrated_classifiers_:
            inner = getattr(cc, "estimator", getattr(cc, "base_estimator", cc))
            boosters.append(_unwrap_pipeline(inner))
        return boosters
    return [_unwrap_pipeline(estimator)]


def _explainers(estimator: Any) -> list:
    """One SHAP TreeExplainer per fold booster."""
    import shap

    return [shap.TreeExplainer(b) for b in _fold_boosters(estimator)]


def _pos_class(vals: Any) -> np.ndarray:
    """Normalise a booster's SHAP output to the positive (distress) class, 2-D."""
    if isinstance(vals, list):
        vals = vals[1] if len(vals) > 1 else vals[0]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1] if vals.shape[2] > 1 else vals[:, :, 0]
    return vals


def _expected_value(explainer: Any) -> float:
    """Scalar base value (positive class) for one fold's explainer."""
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        return float(np.ravel(base)[-1])
    return float(base)


def _prepare(X: pd.DataFrame | np.ndarray, features: Sequence[str]) -> np.ndarray:
    """Feature matrix in training column order.

    NaNs are passed through untouched. The pipeline no longer imputes -- the boosters
    route missing values natively because missingness is informative here (see
    `train.build_pipeline`). Imputing at explain time would describe a different input
    than the model actually scored.
    """
    if isinstance(X, pd.DataFrame):
        return X[list(features)].to_numpy("float64")
    return np.asarray(X, "float64")


def explainer_for(estimator: Any):
    """A SHAP TreeExplainer over the first fold booster.

    Kept for quick single-booster inspection; the scoring paths use the fold-averaged
    `shap_values` / `expected_value`, which describe the whole ensemble.
    """
    return _explainers(estimator)[0]


def shap_values(estimator: Any, X: pd.DataFrame | np.ndarray, features: Sequence[str]) -> np.ndarray:
    """Fold-averaged SHAP values for the positive (distress) class, shape (n_rows, n_features).

    In log-odds space, so bars sum to the ensemble's mean margin (base + sum == mean fold
    margin, exact to floating point). That mean margin is the ranking model behind the
    score, so the waterfall describes the whole ensemble, not one fold that approximates it.
    """
    arr = _prepare(X, features)
    per_fold = [_pos_class(e.shap_values(arr)) for e in _explainers(estimator)]
    return np.mean(per_fold, axis=0)


def expected_value(estimator: Any) -> float:
    """Fold-averaged SHAP base value (positive class) -- the waterfall's starting point."""
    return float(np.mean([_expected_value(e) for e in _explainers(estimator)]))


def contributions(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    features: Sequence[str],
    row: int = 0,
    top_n: int | None = None,
) -> list[Contribution]:
    """Ranked contributions for a single company, largest absolute push first."""
    arr = _prepare(X, features)
    vals = shap_values(estimator, X, features)[row]

    out = [
        Contribution(
            feature=f,
            label=label_for(f),
            value=float(arr[row, i]),
            shap=float(vals[i]),
            category=BY_ID[f].category.value if f in BY_ID else "Other",
        )
        for i, f in enumerate(features)
    ]
    out.sort(key=lambda c: abs(c.shap), reverse=True)
    return out[:top_n] if top_n else out


def contributions_frame(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    features: Sequence[str],
    row: int = 0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Display table: business labels only, raw ids dropped."""
    rows = contributions(estimator, X, features, row=row, top_n=top_n)
    return pd.DataFrame(
        [
            {
                "Driver": c.label,
                "Category": c.category,
                "Value": round(c.value, 4),
                "Contribution": round(c.shap, 4),
                "Effect": c.direction,
            }
            for c in rows
        ]
    )


def global_importance(
    estimator: Any, X: pd.DataFrame | np.ndarray, features: Sequence[str], top_n: int = 15
) -> pd.DataFrame:
    """Mean |SHAP| across companies -- what drives the model overall."""
    vals = np.abs(shap_values(estimator, X, features)).mean(axis=0)
    return (
        pd.DataFrame(
            {
                "Driver": [label_for(f) for f in features],
                "Category": [BY_ID[f].category.value if f in BY_ID else "Other" for f in features],
                "Mean impact": vals.round(5),
            }
        )
        .sort_values("Mean impact", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def waterfall(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    features: Sequence[str],
    row: int = 0,
    max_display: int = 10,
    show: bool = False,
):
    """SHAP waterfall for one company, with labels already renamed.

    Returns the matplotlib figure so the caller can embed it in Streamlit or the PDF.
    """
    import matplotlib.pyplot as plt
    import shap

    arr = _prepare(X, features)
    vals = shap_values(estimator, X, features)
    base = expected_value(estimator)  # fold-averaged, matching the averaged SHAP values

    exp = shap.Explanation(
        values=vals[row],
        base_values=base,
        data=arr[row],
        feature_names=[label_for(f) for f in features],  # never raw Attr names
    )
    shap.plots.waterfall(exp, max_display=max_display, show=show)
    return plt.gcf()


# --------------------------------------------------------------------------------
# Rule-based narrative. Deterministic by design -- see module docstring.
# --------------------------------------------------------------------------------

def _phrase(c: Contribution) -> str:
    """Describe one driver in credit-analyst language."""
    worsening = c.shap > 0
    cat = c.category

    if cat == Category.SOLVENCY.value:
        return f"{'weak' if worsening else 'sound'} {c.label.lower()}"
    if cat == Category.LIQUIDITY.value:
        return f"{'strained' if worsening else 'comfortable'} {c.label.lower()}"
    if cat == Category.PROFITABILITY.value:
        return f"{'depressed' if worsening else 'healthy'} {c.label.lower()}"
    if cat == Category.EFFICIENCY.value:
        return f"{'deteriorating' if worsening else 'stable'} {c.label.lower()}"
    if cat == Category.CASH_FLOW.value:
        return f"{'thin' if worsening else 'solid'} {c.label.lower()}"
    return c.label.lower()


def summarise(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    features: Sequence[str],
    row: int = 0,
    score: float | None = None,
    company: str = "This company",
) -> str:
    """Three-sentence plain-English summary of why the model scored this company.

    Rule-based and fully determined by the SHAP values -- it cannot invent a number.
    """
    top = contributions(estimator, X, features, row=row, top_n=5)
    if not top:
        return f"{company} could not be scored: no model drivers were available."

    risk_drivers = [c for c in top if c.shap > 0][:3]
    supports = [c for c in top if c.shap < 0][:2]

    if score is not None:
        band_word = (
            "critical" if score >= 75 else
            "elevated" if score >= 50 else
            "watch-list" if score >= 25 else
            "healthy"
        )
        first = f"{company} carries a {band_word} risk profile, scoring {score:.0f} out of 100."
    else:
        first = f"{company} has been assessed against the distress model."

    def _join(items: list[Contribution]) -> str:
        phrases = [_phrase(c) for c in items]
        if len(phrases) == 1:
            return phrases[0]
        return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"

    # A low-scoring company must not be narrated by its largest *risk* driver -- that
    # reads as a contradiction ("healthy... the largest contributors are deteriorating
    # X"). Lead with whichever side actually dominates the prediction.
    healthy_profile = score is not None and score < 25

    if healthy_profile and supports:
        second = f"The profile is supported primarily by {_join(supports)}."
        third = (
            f"The main factor working against it is {_phrase(risk_drivers[0])}, "
            "though it is not currently material enough to shift the assessment."
            if risk_drivers
            else "No individual metric is currently pushing this company toward distress."
        )
        return f"{first} {second} {third}"

    if risk_drivers:
        second = f"The largest contributors to this assessment are {_join(risk_drivers)}."
    else:
        second = "No individual metric is currently pushing this company toward distress."

    if supports:
        third = (
            f"Partially offsetting this, {_phrase(supports[0])} "
            f"remains a stabilising factor in the profile."
        )
    elif risk_drivers:
        third = (
            "No material offsetting strengths were identified, so the risk drivers "
            "above are not currently balanced by other financial factors."
        )
    else:
        third = "The financial profile is balanced across the metrics assessed."

    return f"{first} {second} {third}"


@dataclass(frozen=True)
class MonotonicityCheck:
    """Result of sweeping one metric to confirm the score responds correctly.

    `max_magnitude` is the number that matters, not `violation_rate`. A constrained
    booster averaged over folds and passed through sigmoid calibration is monotonic by
    construction, but float accumulation across those layers leaves residue around
    1e-7. Counting those as "violations" is misleading; their *size* is what decides
    whether anything is visible on screen.
    """

    feature: str
    label: str
    violation_rate: float  # fraction of grid steps moving the wrong way
    max_magnitude: float  # largest wrong-direction move, in probability units

    @property
    def max_score_points(self) -> float:
        """Largest wrong move expressed on the displayed 0-100 scale."""
        return self.max_magnitude * 100.0

    @property
    def is_visible(self) -> bool:
        """Would a user ever see this? 0.01 score points is well below one pixel."""
        return self.max_score_points > 0.01


def default_grid(X: pd.DataFrame, feature: str) -> list[float]:
    """Percentile-based sweep values, so checks need no hand-tuned grids."""
    g = np.nanpercentile(X[feature].to_numpy("float64"), [1, 10, 25, 50, 75, 90, 99])
    return sorted(set(float(v) for v in g if np.isfinite(v)))


def monotonicity_violations(
    estimator: Any,
    X: pd.DataFrame,
    features: Sequence[str],
    feature: str,
    grid: Sequence[float] | None = None,
    n_companies: int = 40,
) -> MonotonicityCheck:
    """Sweep one metric and confirm the score moves the economically correct way.

    **This is the demo-safety check.** Any metric exposed as a slider must come back
    with `is_visible == False`, or a user can move that control and watch risk fall as
    the metric worsens. Unconstrained, interest coverage failed on 82.9% of steps.
    """

    d = direction_for(feature)
    if d is Direction.UNCONSTRAINED:
        raise ValueError(
            f"{feature} ({label_for(feature)}) has no economic direction, so "
            "monotonicity is undefined for it"
        )

    if grid is None:
        grid = default_grid(X, feature)

    i = list(features).index(feature)
    idx = np.linspace(0, len(X) - 1, min(n_companies, len(X))).astype(int)
    base = X[list(features)].iloc[idx].to_numpy("float64")

    curves = []
    for v in grid:
        m = base.copy()
        m[:, i] = v
        curves.append(estimator.predict_proba(m)[:, 1])

    steps = np.diff(np.asarray(curves), axis=0)
    # Risk-decreasing features must not increase the score as the value rises.
    wrong = steps > 0 if d is Direction.RISK_DECREASING else steps < 0
    return MonotonicityCheck(
        feature=feature,
        label=label_for(feature),
        violation_rate=float(wrong.mean()),
        max_magnitude=float(np.abs(steps[wrong]).max()) if wrong.any() else 0.0,
    )


def audit_sliders(estimator: Any, X: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Check every user-facing metric at once. Run before any demo."""

    rows = [
        monotonicity_violations(estimator, X, features, f)
        for f in SLIDER_FEATURES
        if f in features
    ]
    return pd.DataFrame(
        [
            {
                "Metric": r.label,
                "Violation rate": f"{r.violation_rate:.2%}",
                "Max wrong move (score pts)": f"{r.max_score_points:.2e}",
                "Visible": r.is_visible,
            }
            for r in rows
        ]
    )


def what_if(
    estimator: Any,
    X: pd.DataFrame,
    features: Sequence[str],
    feature: str,
    new_value: float,
    row: int = 0,
) -> dict[str, float]:
    """Scenario analysis: change one metric, see the score move.

    `feature` accepts either a raw id (`Attr27`) or its business label
    ("Interest Coverage Ratio"), so UI code can pass whatever the user picked.
    """
    resolved = feature
    if feature not in features:
        matches = [f for f in features if label_for(f).lower() == feature.lower()]
        if not matches:
            raise KeyError(f"unknown feature {feature!r}")
        resolved = matches[0]

    base = X[list(features)].iloc[[row]].copy()
    before = float(estimator.predict_proba(base.to_numpy("float64"))[0, 1]) * 100

    changed = base.copy()
    changed.iloc[0, list(features).index(resolved)] = new_value
    after = float(estimator.predict_proba(changed.to_numpy("float64"))[0, 1]) * 100

    return {
        "feature": resolved,
        "label": label_for(resolved),
        "original_value": float(base.iloc[0][resolved]),
        "new_value": float(new_value),
        "score_before": round(before, 1),
        "score_after": round(after, 1),
        "delta": round(after - before, 1),
    }



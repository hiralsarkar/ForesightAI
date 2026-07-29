"""Optuna hyperparameter search (non-negotiable #2).

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
decisions already made against evidence (see AGENTS.md Decision Log) and are not
re-litigated by a random search optimising a single metric. `scale_pos_weight` and
`monotone_constraints` are therefore excluded from the search space -- the builders set
them per-fold, and letting Optuna override them would silently undo the demo-safety
guarantee.

**PR-AUC is the objective**, matching the headline metric. Optimising ROC-AUC at a 3.9%
base rate would tune for the wrong thing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..features.polish_schema import TARGET, Scope
from .train import RANDOM_STATE, cross_validate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "models"

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

    Returns `(TuningResult, study)`. Requires >= 50 trials per non-negotiable #2.
    """
    import optuna

    if n_trials < 50:
        raise ValueError(f"non-negotiable #2 requires >= 50 trials, got {n_trials}")
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

    from .train import BUILDERS, build_pipeline, get_features

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
    """All trials, best first -- the audit trail non-negotiable #2 asks to log."""
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


def save(result: TuningResult, study, name: str | None = None) -> Path:
    """Persist the result and the full trial log."""
    name = name or f"optuna_{result.model_name}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(result.summary(), indent=2), encoding="utf-8"
    )
    trials_frame(study).to_csv(RESULTS_DIR / f"{name}_trials.csv", index=False)
    return RESULTS_DIR / f"{name}.json"

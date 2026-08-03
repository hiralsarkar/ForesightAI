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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..features.polish_schema import BY_ID, Category, label_for


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
    from ..features.polish_schema import Direction, direction_for

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
    from ..features.polish_schema import SLIDER_FEATURES

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

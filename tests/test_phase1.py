"""Phase 1 guardrails.

These tests exist to stop a future session silently undoing a decision that was made
against evidence. Each one maps to an AGENTS.md non-negotiable or trap.

Run: .venv/Scripts/python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import altman, polish_schema as ps
from src.features.load_polish import class_balance, load_horizon
from src.models import explain
from src.models.calibrate import BANDS, band, risk_score
from src.models.train import build_pipeline, get_features, scale_pos_weight


# --------------------------------------------------------------------------- schema
def test_schema_is_complete_and_contiguous():
    ps.validate_schema()  # raises on drift
    assert len(ps.ATTRIBUTES) == 64


def test_every_attribute_has_a_judge_safe_label():
    """Non-negotiable #4: no cryptic variable names on a judge-visible chart."""
    for a in ps.ATTRIBUTES:
        assert a.label != a.id, f"{a.id} has no business label"
        assert not a.label.lower().startswith("attr")
        # Display-cased: may legitimately start with a digit ("3-Year Gross Profit...").
        assert not a.label[0].islower(), f"{a.label!r} is not display-cased"


def test_labels_are_unique():
    labels = [a.label for a in ps.ATTRIBUTES]
    assert len(set(labels)) == len(labels)


def test_serving_set_excludes_scale_dependent_features():
    """Trap 1: absolute currency amounts cannot transfer PLN -> INR."""
    serving = set(ps.serving_feature_set())
    assert "Attr55" not in serving, "absolute working capital must not be served"
    assert ps.SCALE_DEPENDENT.isdisjoint(serving)


def test_serving_set_is_a_strict_subset_of_all_attributes():
    all_ids = {a.id for a in ps.ATTRIBUTES}
    serving = set(ps.serving_feature_set())
    assert serving < all_ids


# --------------------------------------------------------------------------- altman
def test_altman_uses_private_firm_thresholds_not_original_z():
    """Trap 2: 1.81/2.99 require *market* equity and would mis-band these firms."""
    spec = altman.get_spec("z2")
    assert spec.distress_below == 1.10
    assert spec.safe_above == 2.60
    assert spec.distress_below != 1.81, "original-Z threshold applied to a book-equity Z"


def test_altman_returns_nan_when_a_component_is_missing():
    """A Z built from 3 of 4 terms is not a Z."""
    df = pd.DataFrame(
        {"Attr3": [0.1, np.nan], "Attr6": [0.2, 0.2], "Attr7": [0.1, 0.1], "Attr8": [1.0, 1.0]}
    )
    z = altman.altman_z(df, "z2")
    assert not np.isnan(z.iloc[0])
    assert np.isnan(z.iloc[1]), "partial Z must not be reported"


def test_altman_zones_band_correctly():
    z = pd.Series([0.5, 2.0, 5.0])
    zones = altman.zone(z, "z2")
    assert list(zones) == ["Distress", "Grey", "Safe"]


def test_true_bankruptcy_always_labelled_distress():
    """Ground truth outranks the ratio heuristic and is never softened."""
    df = pd.DataFrame(
        {"Attr3": [0.9], "Attr6": [0.9], "Attr7": [0.9], "Attr8": [9.0], "class": [1]}
    )
    # These ratios put the company firmly in the Safe zone...
    assert altman.zone(altman.altman_z(df), "z2").iloc[0] == "Safe"
    # ...but the real label must still win.
    assert altman.three_class_label(df).iloc[0] == altman.CLASS_DISTRESS


# ----------------------------------------------------------------------- pipeline
def test_pipeline_does_not_impute_by_default():
    """Missingness is informative here; imputing it cost 0.014-0.031 PR-AUC."""
    from lightgbm import LGBMClassifier

    pipe = build_pipeline(LGBMClassifier(), use_smote=False)
    assert "impute" not in pipe.named_steps
    assert "smote" not in pipe.named_steps


def test_smote_forces_imputation_because_it_cannot_consume_nan():
    from lightgbm import LGBMClassifier

    pipe = build_pipeline(LGBMClassifier(), use_smote=True)
    assert "impute" in pipe.named_steps
    assert "smote" in pipe.named_steps
    # Order matters: impute must precede SMOTE.
    steps = list(pipe.named_steps)
    assert steps.index("impute") < steps.index("smote")


def test_scale_pos_weight_reflects_imbalance():
    y = np.array([0] * 96 + [1] * 4)
    assert scale_pos_weight(y) == pytest.approx(24.0)


def test_scale_pos_weight_handles_no_positives():
    assert scale_pos_weight(np.zeros(10)) == 1.0


# -------------------------------------------------------------------------- score
def test_risk_score_is_a_faithful_linear_mapping():
    """A displayed 81 must mean an 81% modelled probability."""
    assert risk_score(0.0) == 0.0
    assert risk_score(0.81) == pytest.approx(81.0)
    assert risk_score(1.0) == 100.0


def test_risk_score_clips_out_of_range_input():
    assert risk_score(1.5) == 100.0
    assert risk_score(-0.2) == 0.0


@pytest.mark.parametrize(
    "score,expected",
    [(0, "Healthy"), (24.9, "Healthy"), (25, "Watch"), (49.9, "Watch"),
     (50, "Elevated Risk"), (74.9, "Elevated Risk"), (75, "Critical"), (100, "Critical")],
)
def test_band_boundaries(score, expected):
    assert band(score) == expected


def test_bands_are_contiguous_and_ordered():
    uppers = [u for u, _ in BANDS]
    assert uppers == sorted(uppers)


# --------------------------------------------------------------------------- data
@pytest.fixture(scope="module")
def df1():
    try:
        return load_horizon(1)
    except FileNotFoundError:
        pytest.skip("Polish dataset not downloaded")


def test_horizon_shape_matches_uci_documentation(df1):
    assert len(df1) == 7027
    bal = class_balance(df1)
    assert bal["distress"] == 271
    assert bal["distress_rate"] == pytest.approx(0.0386, abs=1e-3)


def test_majority_baseline_justifies_banning_accuracy(df1):
    """The number quoted when a judge asks about class imbalance."""
    assert class_balance(df1)["majority_baseline_accuracy"] > 0.96


def test_loader_preserves_missingness(df1):
    """If this hits zero, someone added imputation to the loader -- don't."""
    assert df1["Attr37"].isna().mean() > 0.35
    assert df1["Attr27"].isna().sum() > 0


def test_serving_features_all_present_in_data(df1):
    missing = set(get_features(df1, "serving")) - set(df1.columns)
    assert not missing


def test_altman_beats_nothing_but_separates_medians(df1):
    """Z'' signal is real even though its fixed threshold is unusable."""
    z = altman.altman_z(df1, "z2")
    healthy = z[df1["class"] == 0].median()
    distress = z[df1["class"] == 1].median()
    assert healthy > distress * 2


# ------------------------------------------------------------------------ explain
def test_display_labels_never_leak_raw_ids():
    feats = ps.serving_feature_set()
    for name in ps.rename_for_display(feats):
        assert not name.startswith("Attr")


# -------------------------------------------------------------------- monotonicity
def test_directions_are_internally_consistent():
    ps.validate_schema()
    assert ps.direction_for("Attr27") == ps.Direction.RISK_DECREASING  # interest coverage
    assert ps.direction_for("Attr2") == ps.Direction.RISK_INCREASING   # debt to assets


def test_ambiguous_ratios_are_left_unconstrained():
    """A wrong constraint is worse than none -- it encodes false economics silently."""
    for attr in ("Attr9", "Attr21", "Attr29", "Attr44"):  # turnover, growth, size, days
        assert ps.direction_for(attr) == ps.Direction.UNCONSTRAINED


def test_every_slider_feature_has_a_direction():
    """A slider with no direction is exactly the bug this mechanism prevents."""
    for f in ps.SLIDER_FEATURES:
        assert ps.direction_for(f) != ps.Direction.UNCONSTRAINED, f


def test_monotone_vector_aligns_positionally_with_features():
    """Misalignment applies the wrong constraint to the wrong ratio, silently."""
    feats = ["Attr9", "Attr27", "Attr2"]
    assert ps.monotone_vector(feats, "slider") == (0, -1, 1)


def test_monotone_vector_scopes():
    feats = list(ps.serving_feature_set())
    none_v = ps.monotone_vector(feats, "none")
    slider_v = ps.monotone_vector(feats, "slider")
    all_v = ps.monotone_vector(feats, "all")

    assert set(none_v) == {0}
    n_slider = sum(1 for v in slider_v if v != 0)
    n_all = sum(1 for v in all_v if v != 0)
    assert n_slider == len(ps.SLIDER_FEATURES)
    assert n_slider < n_all, "slider scope should constrain strictly fewer features"


def test_monotone_vector_rejects_bad_scope():
    with pytest.raises(ValueError):
        ps.monotone_vector(["Attr1"], "everything")


@pytest.fixture(scope="module")
def fitted(df1):
    from src.models.calibrate import fit_calibrated

    model, _, features = fit_calibrated(df1)
    return model, features, df1[features]


@pytest.mark.slow
@pytest.mark.parametrize("attr", ps.SLIDER_FEATURES)
def test_every_slider_responds_in_the_correct_direction(fitted, attr):
    """End-to-end demo-safety guarantee: no slider may move the score the wrong way.

    Guards the Module 4 what-if and Module 5 stress panels. Unconstrained, interest
    coverage moved the wrong way on 82.9% of steps and Current Ratio on 66.0%.

    Asserted on **magnitude, not violation rate**. The constrained ensemble is monotone
    by construction; what survives is float accumulation across 5 folds plus the sigmoid
    transform, around 1e-7 in probability terms. Counting those as failures would be
    both misleading and brittle. What matters is whether a user could ever see it --
    0.01 score points is far below one pixel of gauge travel.

    Parametrised over the full `SLIDER_FEATURES` set: a slider that isn't covered here
    is a slider that can embarrass us on stage.
    """
    model, features, X = fitted
    check = explain.monotonicity_violations(model, X, features, attr)
    assert not check.is_visible, (
        f"{check.label} moves the wrong way by {check.max_score_points:.4f} score "
        f"points ({check.violation_rate:.1%} of steps) -- visible to a user"
    )


@pytest.mark.slow
def test_slider_audit_covers_every_user_facing_metric(fitted):
    model, features, X = fitted
    audit = explain.audit_sliders(model, X, features)
    assert len(audit) == len(ps.SLIDER_FEATURES)
    assert not audit["Visible"].any()


def test_monotonicity_check_rejects_undirected_features(df1):
    """Asking about an unconstrained ratio is a bug, not a pass."""
    from src.models.train import get_features

    feats = get_features(df1, "serving")
    with pytest.raises(ValueError, match="no economic direction"):
        explain.monotonicity_violations(None, df1[feats], feats, "Attr9")


def test_default_grid_spans_the_observed_range(df1):
    grid = explain.default_grid(df1, "Attr27")
    assert len(grid) >= 5
    assert grid == sorted(grid)


# ------------------------------------------------------------------------- tuning
def test_tuning_enforces_the_fifty_trial_floor():
    """Non-negotiable #2 asks for >= 50 trials; a 5-trial run must not pass as tuned."""
    from src.models import tune as tuning

    with pytest.raises(ValueError, match="50 trials"):
        tuning.tune(None, n_trials=10)


@pytest.mark.parametrize("param", ["scale_pos_weight", "monotone_constraints"])
def test_search_space_cannot_tune_protected_params(param):
    """Letting a random search override these would silently undo evidenced decisions.

    `monotone_constraints` in particular is the demo-safety guarantee -- a search
    optimising PR-AUC alone would happily drop it for +0.02.
    """
    from src.models import tune as tuning

    with pytest.raises(ValueError, match="must not tune"):
        tuning._assert_unprotected({param: 1, "max_depth": 4})


def test_declared_search_spaces_are_clean():
    """The shipped spaces must not contain a protected parameter."""
    from src.models import tune as tuning

    class _Trial:
        def suggest_int(self, name, *a, **k): return 1
        def suggest_float(self, name, *a, **k): return 0.5

    for name, space in tuning.SPACES.items():
        params = space(_Trial())
        assert not (tuning.PROTECTED_PARAMS & set(params)), name


def test_lgbm_subsample_is_paired_with_subsample_freq():
    """LightGBM silently ignores `subsample` unless `subsample_freq` >= 1."""
    from src.models import tune as tuning

    class _Trial:
        def suggest_int(self, name, *a, **k): return 1
        def suggest_float(self, name, *a, **k): return 0.5

    params = tuning.SPACES["lightgbm"](_Trial())
    if "subsample" in params:
        assert params.get("subsample_freq", 0) >= 1


def test_contribution_direction_follows_shap_sign():
    c = explain.Contribution("Attr27", "Interest Coverage Ratio", 0.5, 1.2, "Solvency")
    assert c.direction == "increases risk"
    assert explain.Contribution("Attr27", "x", 0.5, -1.2, "Solvency").direction == "reduces risk"

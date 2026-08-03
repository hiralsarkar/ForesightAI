"""Serving-layer guardrails.

The load-bearing test is `test_acceptance_gate`. If it ever fails, the demo is
broken: some real company scores wrong.

Run: .venv/Scripts/python.exe -m pytest tests/test_serving.py -q
"""

from __future__ import annotations

import math

import pytest

from src.features import altman
from src.serving import demo_companies as demo
from src.serving.financial_score import (
    band_for,
    risk_from_z,
    score_company,
    z_from_risk,
)
from src.serving.screener import (
    ScreenerFinancials,
    compute_features,
    coverage,
    screener_feature_set,
)


# ------------------------------------------------------------------- bridge
def test_bridge_populates_core_features_and_leaves_structural_gaps_nan():
    cov = coverage(demo.VEDANTA_2026)
    # A manufacturer with full days-ratios should populate most of the serving set...
    assert cov["populated"] >= 40
    # ...but the current-ratio family Screener never breaks out stays NaN.
    feats = compute_features(demo.VEDANTA_2026)
    assert math.isnan(feats["Attr4"]), "Current Ratio should be structurally NaN"
    assert math.isnan(feats["Attr46"]), "Quick Ratio should be structurally NaN"


def test_bridge_total_liabilities_excludes_equity():
    """Screener's own 'Total Liabilities' row is the balance-sheet total; the Polish
    attributes mean non-equity claims. Getting this wrong flips leverage ratios."""
    fin = demo.SPICEJET_2026  # negative equity -> tl > total_assets
    assert fin.total_liabilities > fin.total_assets
    feats = compute_features(fin)
    assert feats["Attr2"] > 1.0  # debt/assets exceeds 1 when equity is negative


def test_key_ratios_have_correct_sign_for_distress():
    feats = compute_features(demo.SPICEJET_2026)
    assert feats["Attr10"] < 0   # equity ratio negative (negative net worth)
    assert feats["Attr27"] < 0   # interest coverage negative (operating loss)
    assert feats["Attr2"] > 1.0  # liabilities exceed assets


def test_sales_growth_needs_prior_year():
    with_prior = compute_features(demo.SPICEJET_2026, prior=demo.VEDANTA_2026)
    without = compute_features(demo.SPICEJET_2026)
    assert not math.isnan(with_prior["Attr21"])
    assert math.isnan(without["Attr21"])


def test_screener_feature_set_is_subset_of_serving():
    from src.features.polish_schema import serving_feature_set

    assert set(screener_feature_set()) <= set(serving_feature_set())
    assert 40 <= len(screener_feature_set()) <= 50


# --------------------------------------------------------- financial score
def test_risk_score_anchors_on_altman_zone_thresholds():
    """Band boundaries must coincide with Altman's own zones, exactly."""
    assert risk_from_z(altman.Z_DOUBLE_PRIME.safe_above) == pytest.approx(25.0, abs=0.1)
    assert risk_from_z(altman.Z_DOUBLE_PRIME.distress_below) == pytest.approx(50.0, abs=0.1)


def test_risk_score_is_monotonic_decreasing_in_z():
    zs = [-20, -5, 0, 1.1, 2.6, 5, 11]
    risks = [risk_from_z(z) for z in zs]
    assert risks == sorted(risks, reverse=True)
    assert 0.0 <= min(risks) and max(risks) <= 100.0


def test_risk_and_z_inverse_round_trip():
    for risk in (10, 25, 50, 75, 90):
        assert risk_from_z(z_from_risk(risk)) == pytest.approx(risk, abs=0.1)


def test_decomposition_terms_sum_to_z_exactly():
    """The waterfall is exact -- no approximation, unlike SHAP."""
    s = score_company(demo.SPICEJET_2026)
    assert sum(t.contribution for t in s.terms) == pytest.approx(s.z_score, abs=1e-9)


def test_missing_working_capital_is_flagged_not_silent():
    bare = ScreenerFinancials(
        "NoWC", 2019, sales=1000, expenses=800, operating_profit=200, other_income=10,
        interest=10, depreciation=20, profit_before_tax=200, net_profit=150,
        equity_capital=50, reserves=400, borrowings=100, other_liabilities=450,
        total_assets=1000, fixed_assets=300,  # no working_capital_days
    )
    s = score_company(bare)
    assert s.missing_terms  # WC term recorded
    assert not s.is_reliable
    assert not math.isnan(s.z_score)  # still computes (WC filled 0)


@pytest.mark.parametrize("risk,band", [
    (0, "Healthy"), (24, "Healthy"), (25, "Watch"), (49, "Watch"),
    (50, "Elevated Risk"), (74, "Elevated Risk"), (75, "Critical"), (100, "Critical"),
])
def test_bands_match_the_calibrate_module(risk, band):
    from src.models.calibrate import band as ml_band

    assert band_for(risk) == band
    assert ml_band(risk) == band  # financial and ML scores speak one language


# ------------------------------------------------------- THE acceptance gate
def test_acceptance_gate():
    """Every real demo company scores in its expected band.

    This is the test that proves the serving architecture works. Do not weaken the
    expectations to make it pass -- fix the score.
    """
    results = demo.validate_roster()
    failures = [r for r in results if not r["pass"]]
    assert not failures, "roster mis-scored: " + ", ".join(
        f"{r['company']}={r['band']} (wanted {r['expected']})" for r in failures
    )


def test_gate_covers_both_extremes_and_a_watch_case():
    expects = {r["expected"] for r in demo.validate_roster()}
    assert {"healthy", "distress", "watch"} <= expects


def test_healthy_controls_are_clearly_healthy():
    """Not just under 25 -- the controls should have real headroom, or a small data
    change could tip them into Watch on the day."""
    results = {r["company"]: r for r in demo.validate_roster()}
    for name in ("TCS", "Paytm"):
        assert results[name]["risk"] < 15, f"{name} too close to the Watch line"

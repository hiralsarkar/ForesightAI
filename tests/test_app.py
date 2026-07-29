"""Dashboard scoring-service guardrails (Module 5 stress path especially).

The stress sliders are a demo centerpiece, and a dead slider is the exact fatal-demo
moment monotonicity was built to prevent. These tests exercise both stress legs headless
so a regression (e.g. a shock that moves none of the Altman terms) fails CI.
"""

from __future__ import annotations

import pytest

from app import scoring_service as svc


def test_roster_has_expected_companies():
    names = svc.company_names()
    assert "SpiceJet" in names and "Vedanta" in names
    assert len(names) == 6


def test_ebit_shock_leg_moves_the_score():
    """A fall in operating profit must raise risk."""
    assert svc.stress("Vedanta", -30, 0).delta > 1.0
    # And a rise should lower it.
    assert svc.stress("Vedanta", 20, 0).delta < 0


def test_leverage_shock_leg_is_not_dead():
    """Regression: the borrowings slider once moved none of the Altman terms because
    the shock touched `borrowings` but not `total_assets`. It must bite on a leveraged
    company, and in the right direction."""
    fr = svc.stress("Vedanta", 0, 40)
    assert fr.delta > 1.0, "leverage slider is dead on a leveraged company"
    # Less debt is safer.
    assert svc.stress("Vedanta", 0, -20).delta < 0


def test_leverage_direction_is_correct_everywhere():
    """More debt never lowers risk, for any company."""
    for co in svc.company_names():
        up = svc.stress(co, 0, 40).delta
        assert up >= -0.05, f"{co}: adding debt lowered risk ({up})"


def test_stress_uses_altman_not_gbm():
    """The stress result exposes the Altman Z, confirming the linear (instant) path."""
    r = svc.stress("Vedanta", -30, 40)
    assert r.base_z != r.stressed_z
    assert r.stressed_z < r.base_z  # worse scenario -> lower Z


def test_combined_uses_both_legs_for_every_company():
    for name in svc.company_names():
        c = svc.combined(name)
        assert c.has_digital and c.digital_weight > 0, name


# ------------------------------------------------------------------ portfolio (M8)
def test_portfolio_covers_roster_sorted_worst_first():
    rows = svc.portfolio()
    assert len(rows) == len(svc.company_names())
    scores = [r.combined for r in rows]
    assert scores == sorted(scores, reverse=True), "portfolio must lead with highest risk"


def test_portfolio_separates_distress_from_healthy():
    """The visual contrast that makes the surveillance view worth looking at."""
    rows = {r.company: r for r in svc.portfolio()}
    assert rows["SpiceJet"].band == "Critical"
    assert rows["Ola Electric"].band == "Critical"
    for healthy in ("TCS", "Paytm"):
        assert rows[healthy].band == "Healthy", healthy


def test_every_roster_company_has_complete_signal_coverage():
    """The demo roster is chosen so no company has a data gap."""
    for r in svc.portfolio():
        assert r.has_digital, r.company
        assert len(svc.digital(r.company).readings) == 4, r.company


# ------------------------------------------------------------- Jet timeline (case study)
def test_case_timeline_is_chronological_and_deteriorates():
    tl = svc.case_timeline()
    assert len(tl) >= 4
    assert tl[-1].digital > tl[0].digital, "digital should worsen toward the collapse"
    assert tl[-1].digital >= 55, "digital should be elevated by the latest point"


def test_case_financial_line_is_the_annual_filing():
    """The financial score is a single annual figure -- flat across the window by design,
    which is exactly the contrast the case study draws against the live signals."""
    tl = svc.case_timeline()
    assert len({p.financial for p in tl}) == 1


def test_default_demo_company_moves_on_both_sliders():
    """The first-load company must make Module 5 look alive."""
    from app.scoring_service import stress

    assert stress("Vedanta", -30, 0).delta > 1.0   # EBIT leg
    assert stress("Vedanta", 0, 40).delta > 1.0     # leverage leg


# ------------------------------------------------------- macro stress (M5)
def test_macro_adverse_scenario_raises_risk_for_every_company():
    """Worse macro (rates up, inflation up, GDP down) must never lower risk -- the
    direction discipline that caught the dead leverage slider, now for the macro path."""
    for co in svc.company_names():
        r = svc.stress(co, interest_bps=300, inflation_pp=4, gdp_pp=-3)
        assert r.delta >= -0.05, f"{co}: adverse macro lowered risk ({r.delta})"


def test_each_macro_lever_is_alive_and_correctly_signed():
    from src.scoring.stress import Scenario, run
    from src.serving.demo_companies import ROSTER, SECTORS
    # Use a mid-range, levered company where the logistic is steep.
    rec, prior, _ = next(r for r in ROSTER if r[0].company == "Vedanta")
    prof_sector = SECTORS["Vedanta"]
    base = run(rec, prof_sector, Scenario(), prior).base_score
    for sc in (Scenario(interest_bps=300), Scenario(inflation_pp=4), Scenario(gdp_pp=-3)):
        assert run(rec, prof_sector, sc, prior).stressed_score > base


def test_interest_shock_worsens_coverage():
    """A rate rise must show up in interest coverage -- the headline credit channel."""
    r = svc.stress("Vodafone Idea", interest_bps=300)
    assert r.stressed_coverage < r.base_coverage


def test_rate_shock_leaves_ebit_terms_untouched_hits_equity():
    """Interest sits below EBIT: a pure rate shock must not change the EBIT/TA term, only
    erode equity via retained earnings. Guards the economics."""
    from src.scoring.stress import Scenario, apply_to_financials, sector_profile
    from src.serving.demo_companies import VEDANTA_2026
    shocked, extra = apply_to_financials(VEDANTA_2026, Scenario(interest_bps=300),
                                         sector_profile("Metals & Mining"))
    assert shocked.operating_profit == VEDANTA_2026.operating_profit  # EBIT untouched
    assert shocked.reserves < VEDANTA_2026.reserves                   # equity eroded
    assert extra > 0


def test_sector_elasticities_differ_it_vs_airline():
    """IT has pricing power (low inflation hit); an airline does not. The model must
    reflect that, or the sector story is indefensible."""
    from src.scoring.stress import sector_profile
    assert sector_profile("IT Services").inflation_beta < sector_profile("Airline").inflation_beta

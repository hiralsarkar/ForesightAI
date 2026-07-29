"""Module 3 fusion guardrails.

The two load-bearing properties: financial-only companies must renormalize (never a
phantom digital=0 term), and the narrative must not claim a divergence the data doesn't
show.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.scoring.combined import fuse
from src.serving.demo_companies import OLA_2026, SPICEJET_2026, TCS_2026, VEDANTA_2026
from src.serving.financial_score import score_company
from src.signals.demo_signals import DEFAULT_AS_OF, pulse_as_of
from src.signals.sentiment import LoughranMcDonaldScorer


def _fin(rec, prior=None):
    return score_company(rec, prior=prior)


def _dig(name):
    return pulse_as_of(name, DEFAULT_AS_OF[name], LoughranMcDonaldScorer())


def test_financial_only_renormalizes_to_full_weight():
    """No phantom digital=0 term -- that would drag a distressed company toward healthy."""
    c = fuse(_fin(TCS_2026), digital=None)
    assert not c.has_digital
    assert c.financial_weight == 1.0
    assert c.digital_weight == 0.0
    assert c.combined_score == pytest.approx(c.financial_score, abs=0.1)
    assert "financial signals alone" in c.narrative or "financials only" in c.narrative


def test_distressed_financial_only_stays_distressed():
    """The bias check: a distressed company with no digital data must not soften."""
    c = fuse(_fin(SPICEJET_2026), digital=None)
    assert c.combined_score > 75  # Critical, not dragged down by a missing-digital term


def test_fusion_is_weighted_average_when_both_present():
    fs = _fin(TCS_2026)
    dg = _dig("TCS")
    c = fuse(fs, dg, financial_weight=0.6, digital_weight=0.4)
    expected = 0.6 * fs.risk_score + 0.4 * dg.composite_score
    assert c.combined_score == pytest.approx(expected, abs=0.1)
    assert c.digital_weight == pytest.approx(0.4)


def test_weights_renormalize_if_not_summing_to_one():
    fs, dg = _fin(TCS_2026), _dig("TCS")
    c = fuse(fs, dg, financial_weight=3.0, digital_weight=1.0)  # 3:1
    assert c.financial_weight == pytest.approx(0.75)
    assert c.digital_weight == pytest.approx(0.25)


def test_narrative_matches_the_actual_band_relationship():
    """Never claim a divergence the bands don't show, nor miss one they do."""
    for name, rec, prior in [("TCS", TCS_2026, None),
                             ("SpiceJet", SPICEJET_2026, None),
                             ("Vedanta", VEDANTA_2026, None)]:
        c = fuse(_fin(rec, prior), _dig(name))
        from src.serving.financial_score import band_for
        same_band = band_for(c.financial_score) == band_for(c.digital_score)
        assert ("agree" in c.narrative) == same_band, name


def test_divergence_narrative_fires_only_on_band_disagreement():
    """Synthetic check that the flagship divergence text is reachable -- but only when the
    legs actually fall in different bands, never for a within-band gap."""
    from src.serving.financial_score import score_company
    from src.signals.composite import DigitalPulse

    healthy_fin = score_company(TCS_2026)  # Healthy (~0)
    # Fabricate a digital pulse in a worse band to exercise the branch.
    bad_digital = DigitalPulse(company="TCS", as_of=date(2019, 3, 25), readings=[], composite_score=70.0)
    c = fuse(healthy_fin, bad_digital)
    assert "divergence" in c.narrative
    assert "look closer" in c.narrative


def test_combined_band_matches_score():
    c = fuse(_fin(SPICEJET_2026), _dig("SpiceJet"))
    assert c.band == "Critical"

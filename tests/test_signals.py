"""Module 2 digital-signals guardrails.

The load-bearing test is point-in-time integrity: a past assessment must never see a
future filing.

Run: .venv/Scripts/python.exe -m pytest tests/test_signals.py -q
"""

from __future__ import annotations

from datetime import date

import pytest

from src.signals import demo_signals as demo
from src.signals.base import SignalKind, SignalReading, Trend, band_for
from src.signals.composite import combine
from src.signals.hiring import HiringObservation
from src.signals.hiring import score_as_of as hiring_score
from src.signals.leadership import EventType, LeadershipEvent, Role
from src.signals.leadership import score_as_of as leadership_score
from src.signals.sentiment import (
    Headline,
    LoughranMcDonaldScorer,
    score_as_of as sentiment_score,
)


# -------------------------------------------------------- point-in-time
def test_leadership_is_point_in_time():
    """An assessment must not see a resignation filed after its date."""
    events = demo.OLA["leadership"]
    early = leadership_score(events, date(2025, 10, 1), "Ola Electric")  # before any exit
    late = leadership_score(events, date(2026, 5, 31), "Ola Electric")   # after all exits
    assert early.raw == 0
    assert late.raw == len(events)  # all curated exits visible by the crisis date
    assert late.risk_score > early.risk_score


def test_pulse_as_of_never_leaks_future_events():
    """Whole-pulse point-in-time check on the headline case."""
    before = demo.pulse_as_of("Ola Electric", date(2025, 7, 1))
    after = demo.pulse_as_of("Ola Electric", date(2026, 5, 31))
    assert after.composite_score > before.composite_score
    lead_before = before.by_kind(SignalKind.LEADERSHIP)
    assert lead_before.raw == 0  # no exits filed by mid-2025


# ------------------------------------------------------------------- leadership
def test_more_than_two_exits_is_flagged_red():
    """Blueprint rule: > 2 senior exits -> Elevated/Critical."""
    events = [
        LeadershipEvent("X", date(2019, 1, 1), Role.CEO, "a", EventType.RESIGNATION),
        LeadershipEvent("X", date(2019, 2, 1), Role.CFO, "b", EventType.RESIGNATION),
        LeadershipEvent("X", date(2019, 3, 1), Role.INDEPENDENT_DIRECTOR, "c", EventType.RESIGNATION),
    ]
    r = leadership_score(events, date(2019, 3, 15))
    assert r.risk_score >= 50  # Elevated or worse
    assert r.label == "High turnover"


def test_auditor_exit_weighted_higher_than_independent_director():
    aud = leadership_score(
        [LeadershipEvent("X", date(2019, 1, 1), Role.AUDITOR, "a", EventType.AUDITOR_EXIT)],
        date(2019, 2, 1),
    )
    ind = leadership_score(
        [LeadershipEvent("X", date(2019, 1, 1), Role.INDEPENDENT_DIRECTOR, "a", EventType.RESIGNATION)],
        date(2019, 2, 1),
    )
    assert aud.risk_score > ind.risk_score


def test_appointments_do_not_raise_risk():
    r = leadership_score(
        [LeadershipEvent("X", date(2019, 1, 1), Role.CEO, "a", EventType.APPOINTMENT)],
        date(2019, 2, 1),
    )
    assert r.raw == 0  # appointments are not exits


# -------------------------------------------------------------------- sentiment
def test_lm_scorer_reads_financial_negatives():
    s = LoughranMcDonaldScorer()
    assert s.score("Company defaults on loan amid liquidity crisis") < 0
    assert s.score("Company posts record profit and strong growth") > 0
    assert s.score("Company holds annual general meeting") == 0.0


def test_sentiment_risk_rises_as_tone_worsens():
    neg = [Headline(date(2019, 3, 10), "loss default crisis restructuring downgrade")]
    pos = [Headline(date(2019, 3, 10), "profit growth record strong upgrade")]
    r_neg = sentiment_score(neg, date(2019, 3, 20))
    r_pos = sentiment_score(pos, date(2019, 3, 20))
    assert r_neg.risk_score > r_pos.risk_score


# -------------------------------------------------------------------- hiring
def test_hiring_contraction_raises_risk():
    obs = [HiringObservation(date(2018, 6, 1), 300), HiringObservation(date(2019, 3, 1), 80)]
    r = hiring_score(obs, date(2019, 3, 15))
    assert r.label == "Contracting"
    assert r.risk_score > 50


# -------------------------------------------------------------------- composite
def test_composite_renormalises_over_missing_signals():
    """A company with only two signals is not penalised for the two it lacks."""
    readings = [
        SignalReading(SignalKind.LEADERSHIP, date(2019, 1, 1), 80, "x", "d"),
        SignalReading(SignalKind.NEWS_SENTIMENT, date(2019, 1, 1), 80, "x", "d"),
    ]
    p = combine("X", date(2019, 1, 1), readings)
    assert p.composite_score == pytest.approx(80.0, abs=0.1)


def test_composite_bands_match_financial_scale():
    from src.serving.financial_score import band_for as fin_band

    for score in (10, 40, 60, 90):
        assert band_for(score) == fin_band(score)


# ---------------------------------------------------------------- the digital gate
def test_default_scorer_is_always_available():
    """With or without torch, there must be a working scorer, so the demo keeps running."""
    from src.signals.sentiment import default_scorer

    s = default_scorer()
    assert -1.0 <= s.score("Company defaults on loan") <= 1.0
    # Fallback path must return the lexicon scorer, not raise.
    assert default_scorer(prefer_finbert=False).name == "Loughran-McDonald"


@pytest.mark.slow
def test_finbert_agrees_with_lexicon_on_sign_but_adds_nuance():
    """FinBERT earns the torch dependency: it scores context L-M reads as neutral."""
    import importlib.util

    if not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")):
        pytest.skip("torch/transformers not installed")

    from src.signals.sentiment import FinBertScorer, LoughranMcDonaldScorer

    fb, lm = FinBertScorer(), LoughranMcDonaldScorer()
    # A headline the lexicon misses (no lexicon words) but is clearly negative.
    missed = "Jet Airways defers loan repayment, lenders in talks"
    assert lm.score(missed) == 0.0
    assert fb.score(missed) < -0.1  # FinBERT reads the context


# ------------------------------------------------- verified data & coverage
def test_every_reading_has_a_specific_datum():
    """No gauge may show a bare metric -- the datum is required and must be non-trivial."""
    for co in ("TCS", "Ola Electric", "Vedanta"):
        for r in demo.pulse_as_of(co, demo.DEFAULT_AS_OF[co]).readings:
            assert len(r.datum) > 20

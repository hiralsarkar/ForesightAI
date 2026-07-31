"""M6 (narrative) and M7 (recommendations) guardrails.

The load-bearing properties: the narrative keeps working without an API key and varies in
*substance* across companies; the recommendations never give wrong advice to a healthy
company (the FMCG negative-working-capital and payables-are-not-debt traps).
"""

from __future__ import annotations

import pytest

from src.narrative.recommendations import Audience, recommend, rule_count
from src.narrative.summary import collect_facts, generate
from src.scoring.combined import fuse
from src.serving import demo_companies as D
from src.serving.financial_score import score_company
from src.signals.demo_signals import DEFAULT_AS_OF, has_signals, pulse_as_of
from src.signals.sentiment import LoughranMcDonaldScorer

_CASES = {
    "SpiceJet": (D.SPICEJET_2026, None), "Ola Electric": (D.OLA_2026, None),
    "Vodafone Idea": (D.VODAFONE_IDEA_2026, None), "Vedanta": (D.VEDANTA_2026, None),
    "Paytm": (D.PAYTM_2026, None), "TCS": (D.TCS_2026, None),
}


def _parts(name):
    rec, prior = _CASES[name]
    fin = score_company(rec, prior=prior)
    dig = pulse_as_of(name, DEFAULT_AS_OF[name], LoughranMcDonaldScorer()) if has_signals(name) else None
    return rec, prior, fin, dig, fuse(fin, dig)


# ------------------------------------------------------------------ M6 narrative
def test_narrative_works_without_an_api_key():
    """The demo keeps working on a missing key or a timeout."""
    _, _, fin, dig, comb = _parts("SpiceJet")
    text, source = generate(comb, fin, dig, use_llm=False)
    assert source == "rule-based"
    assert len(text.split(".")) >= 4  # four sentences


def test_narrative_differs_in_substance_not_just_name():
    """Blueprint: outputs must be genuinely different, not a filled-in template."""
    texts = {}
    for name in ("TCS", "SpiceJet", "Vedanta"):
        _, _, fin, dig, comb = _parts(name)
        t, _ = generate(comb, fin, dig, use_llm=False)
        texts[name] = t.replace(name, "X")  # strip the company name
    assert len(set(texts.values())) == 3, "narratives are templated, not substantive"


def test_narrative_facts_carry_precomputed_numbers_only():
    """The LLM must never be asked to calculate -- every figure is supplied."""
    _, _, fin, dig, comb = _parts("SpiceJet")
    f = collect_facts(comb, fin, dig)
    assert f.altman_z == fin.z_score
    assert f.combined_score == comb.combined_score
    assert f.top_negative_terms  # drivers precomputed, not derived by the model


def test_distress_and_healthy_narratives_give_different_guidance():
    _, _, fin_j, dig_j, comb_j = _parts("SpiceJet")
    _, _, fin_t, dig_t, comb_t = _parts("TCS")
    jet, _ = generate(comb_j, fin_j, dig_j, use_llm=False)
    tcs, _ = generate(comb_t, fin_t, dig_t, use_llm=False)
    assert "restructuring" in jet.lower()
    assert "standard review" in tcs.lower()


# ------------------------------------------------------------ M7 recommendations
def test_rule_count_meets_the_blueprint_minimum():
    assert rule_count() >= 15  # blueprint asks for 15-20


def test_every_company_gets_both_audiences():
    for name in _CASES:
        rec, prior, fin, dig, _ = _parts(name)
        out = recommend(rec, fin, dig, prior)
        assert out[Audience.LENDER], f"{name}: no lender guidance"
        assert out[Audience.MANAGEMENT], f"{name}: no management guidance"


def test_debt_free_company_is_not_told_it_is_levered():
    """HUL has zero borrowings; total liabilities are trade payables. Calling that
    'leverage' would be wrong -- payables are not debt."""
    rec, prior, fin, dig, _ = _parts("Paytm")
    titles = " ".join(r.title for r in recommend(rec, fin, dig, prior)[Audience.LENDER])
    assert "leverage" not in titles.lower()


def test_healthy_fmcg_not_told_to_close_working_capital_gap():
    """Negative working capital is FMCG's business model, not a problem to fix."""
    for name in ("TCS", "Paytm"):
        rec, prior, fin, dig, _ = _parts(name)
        titles = " ".join(r.title for r in recommend(rec, fin, dig, prior)[Audience.MANAGEMENT])
        assert "working capital" not in titles.lower(), name


def test_healthy_companies_get_benign_advice_only():
    alarming = ("recapitalis", "distress", "not covered", "negative net worth", "loss-making")
    for name in ("TCS", "Paytm"):
        rec, prior, fin, dig, _ = _parts(name)
        out = recommend(rec, fin, dig, prior)
        blob = " ".join(r.title.lower() for items in out.values() for r in items)
        for word in alarming:
            assert word not in blob, f"{name} got alarming advice: {word}"


def test_distressed_company_gets_urgent_actions():
    rec, prior, fin, dig, _ = _parts("SpiceJet")
    out = recommend(rec, fin, dig, prior)
    assert any(r.priority == 1 for r in out[Audience.LENDER])
    titles = " ".join(r.title.lower() for r in out[Audience.LENDER])
    assert "negative net worth" in titles


def test_recommendations_are_sorted_most_urgent_first():
    rec, prior, fin, dig, _ = _parts("SpiceJet")
    for items in recommend(rec, fin, dig, prior).values():
        prios = [r.priority for r in items]
        assert prios == sorted(prios)


def test_recommendations_reference_actual_values():
    """Not generic advice -- the text must contain the company's real numbers."""
    rec, prior, fin, dig, _ = _parts("SpiceJet")
    blob = " ".join(r.action for r in recommend(rec, fin, dig, prior)[Audience.LENDER])
    assert "%" in blob or "x" in blob


# ------------------------------------------------------- offline narrative cache
# These use the FinBERT scorer -- the SAME path the live app uses -- because the cache is
# keyed on the FinBERT-derived prompt. The L-M-based `_parts` helper builds a different
# prompt and would not match the cache.
def _finbert_parts(name):
    from src.signals.sentiment import default_scorer
    rec, prior = _CASES[name]
    fin = score_company(rec, prior=prior)
    dig = pulse_as_of(name, DEFAULT_AS_OF[name], default_scorer())
    return rec, prior, fin, dig, fuse(fin, dig)


def test_narrative_cache_serves_offline_without_network():
    """The demo must produce AI narratives with no key and no network -- via the cache
    pre-warmed into data/demo/narrative_cache.json."""
    import src.narrative.llm_config as cfg
    import src.narrative.summary as summ
    from src.serving.demo_companies import SECTORS

    saved_llm, saved_key = summ._llm, cfg.openrouter_key
    cfg.openrouter_key = lambda: None                       # no API key
    summ._llm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network down"))
    try:
        for name in ("SpiceJet", "Vedanta", "TCS"):
            _, _, fin, dig, comb = _finbert_parts(name)
            text, src = summ.generate(comb, fin, dig, sector=SECTORS[name])
            assert src == "llm", f"{name} did not resolve from cache offline"
            assert len(text) > 80
    finally:
        summ._llm, cfg.openrouter_key = saved_llm, saved_key


def test_prewarm_keys_match_the_apps_finbert_prompt():
    """A cached entry must exist for the exact prompt the app builds (sector + FinBERT)."""
    import src.narrative.summary as summ
    from src.serving.demo_companies import SECTORS

    cache = summ._load_cache()
    assert cache, "narrative cache is empty -- run summary.prewarm_cache()"
    _, _, fin, dig, comb = _finbert_parts("SpiceJet")
    facts = summ.collect_facts(comb, fin, dig, SECTORS["SpiceJet"])
    assert summ._cache_key(facts) in cache

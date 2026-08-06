"""Cached scoring layer between the Streamlit UI and the `src` engines.

Streamlit reruns the whole script on every interaction, so anything expensive must be
cached or the stress sliders miss the <2s bar:

* `@st.cache_resource` -- the GBM artefact and FinBERT, loaded once per process.
* `@st.cache_data` -- per-company scores, computed once and memoised.

The stress-test path deliberately routes through Altman (a linear formula, instant), so a
slider drag never touches the GBM. The GBM is only loaded for the optional methodology
panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import streamlit as st

from foresight import CombinedRisk, fuse
import foresight as demo
from foresight import FinancialScore, risk_from_z, score_company
from foresight import ScreenerFinancials, compute_features
from foresight import DigitalPulse
from foresight import DEFAULT_AS_OF, has_signals, pulse_as_of

# Roster: display name -> (record, prior record or None)
_ROSTER: dict[str, tuple[ScreenerFinancials, Optional[ScreenerFinancials]]] = {
    "SpiceJet": (demo.SPICEJET_2026, None),
    "Ola Electric": (demo.OLA_2026, None),
    "Vodafone Idea": (demo.VODAFONE_IDEA_2026, None),
    "Vedanta": (demo.VEDANTA_2026, None),
    "Paytm": (demo.PAYTM_2026, None),
    "TCS": (demo.TCS_2026, None),
}


def company_names() -> list[str]:
    return list(_ROSTER)


def get_record(name: str) -> tuple[ScreenerFinancials, Optional[ScreenerFinancials]]:
    return _ROSTER[name]


@st.cache_resource(show_spinner=False)
def _sentiment_scorer():
    """FinBERT if available, else the lexicon. Loaded once per process."""
    from foresight import default_scorer

    return default_scorer()


@st.cache_data(show_spinner=False)
def financial(name: str) -> FinancialScore:
    rec, prior = _ROSTER[name]
    return score_company(rec, prior=prior)


@st.cache_data(show_spinner=False)
def digital(name: str) -> Optional[DigitalPulse]:
    if not has_signals(name):
        return None
    return pulse_as_of(name, DEFAULT_AS_OF[name], _sentiment_scorer())


@st.cache_data(show_spinner=False)
def combined(name: str) -> CombinedRisk:
    return fuse(financial(name), digital(name))


@st.cache_data(show_spinner=False)
def distress_probability(name: str) -> Optional[float]:
    """Learned P(distress) from the India-trained logistic model. None if unavailable."""
    rec, prior = _ROSTER[name]
    return demo.india_distress_probability(rec, prior=prior)


@st.cache_data(show_spinner=False)
def live_llm_sentiment(name: str, heads: tuple) -> Optional[dict]:
    """Magnitude-aware LLM sentiment for a live company's headlines; cached per headline set.
    Returns None when the LLM is unavailable, so the caller keeps the lexicon score."""
    return demo.llm_news_sentiment(name, list(heads))


@st.cache_data(show_spinner=False)
def live_analyst_summary(name: str, altman_z: float, zone: str, combined_score: float,
                         band: str, term_pairs: tuple, news_risk: float,
                         news_rationale: str, model_prob) -> Optional[str]:
    """Cached LLM analyst summary for a live company. None if the LLM is unavailable."""
    return demo.live_analyst_summary(name, altman_z, zone, combined_score, band,
                                     list(term_pairs), news_risk, news_rationale, model_prob)


# Sector labels -- single source of truth in demo_companies so the UI and the narrative
# pre-warm produce identical prompts (and therefore identical cache keys).
from foresight import SECTORS as _SECTOR


@dataclass
class PortfolioRow:
    company: str
    sector: str
    combined: float
    financial: float
    digital: Optional[float]
    band: str
    has_digital: bool


@st.cache_data(show_spinner=False)
def portfolio() -> list[PortfolioRow]:
    """Score every company for the surveillance table, worst risk first."""
    rows = []
    for name in company_names():
        c = combined(name)
        rows.append(PortfolioRow(
            company=name, sector=_SECTOR.get(name, ""),
            combined=c.combined_score, financial=c.financial_score,
            digital=c.digital_score, band=c.band, has_digital=c.has_digital,
        ))
    rows.sort(key=lambda r: r.combined, reverse=True)
    return rows


# -------------------------------------------------- narrative + advice
@st.cache_data(show_spinner=False)
def narrative(name: str) -> tuple[str, str]:
    """AI narrative + its source ('llm' or 'rule-based'). Never raises."""
    from foresight import generate

    return generate(combined(name), financial(name), digital(name), _SECTOR.get(name, ""))


@st.cache_data(show_spinner=False)
def advice(name: str):
    """Rule-based recommendations grouped by audience."""
    from foresight import recommend

    rec, prior = _ROSTER[name]
    return recommend(rec, financial(name), digital(name), prior)


@st.cache_data(show_spinner=False)
def report_pdf(name: str) -> bytes:
    """Two-page executive memo. Cached so repeat clicks are instant."""
    from foresight import build_report

    rec, prior = _ROSTER[name]
    text, _src = narrative(name)
    return build_report(rec, combined(name), financial(name), digital(name),
                        text, advice(name), _SECTOR.get(name, ""), prior)


# ------------------------------------------------ Ola Electric case-study timeline
@dataclass
class TimelinePoint:
    label: str
    digital: float
    financial: float


@st.cache_data(show_spinner=False)
def case_timeline(name: str = "Ola Electric") -> list[TimelinePoint]:
    """Digital Pulse over the last year against the financial score.

    Ola Electric's deterioration is documented month by month through FY2026 -- the CFO
    and two other C-suite exits, a 5% workforce cut, and three straight months of falling
    registrations -- while the financial score reflects the single FY2026 annual filing.
    That contrast is the point: financials are an annual snapshot, the signals are live.
    """
    from datetime import date as _d

    from foresight import pulse_as_of as _pulse

    scorer = _sentiment_scorer()
    fin = financial(name).risk_score
    points = [("Jun 2025", _d(2025, 6, 30)), ("Sep 2025", _d(2025, 9, 30)),
              ("Dec 2025", _d(2025, 12, 31)), ("Mar 2026", _d(2026, 3, 31)),
              ("Jun 2026", _d(2026, 6, 30))]
    return [TimelinePoint(label=l, digital=_pulse(name, w, scorer).composite_score,
                          financial=fin) for l, w in points]


# --------------------------------------------------------------- financial ratios
@dataclass
class RatioCard:
    label: str
    value: str
    context: str
    good: bool


def _fmt(x: float, pct: bool = False, suffix: str = "") -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x*100:.0f}%" if pct else f"{x:.2f}{suffix}"


@st.cache_data(show_spinner=False)
def ratio_cards(name: str) -> list[RatioCard]:
    """Key financial ratios with one line of context each (tracked roster)."""
    rec, prior = _ROSTER[name]
    return _cards_from_features(compute_features(rec, prior=prior))


def ratio_cards_fin(fin) -> list[RatioCard]:
    """The same ratio cards for any live-fetched company."""
    return _cards_from_features(compute_features(fin))


def _cards_from_features(f) -> list[RatioCard]:
    cards = []
    # Interest coverage
    ic = f.get("Attr27", float("nan"))
    cards.append(RatioCard(
        "Interest Coverage", _fmt(ic, suffix="x"),
        "Operating profit vs interest due. Below 1x means operating income cannot cover interest.",
        ic == ic and ic >= 2,
    ))
    # Debt to assets
    da = f.get("Attr2", float("nan"))
    cards.append(RatioCard(
        "Debt to Assets", _fmt(da, pct=True),
        "Share of assets funded by liabilities. Above 100% means negative net worth.",
        da == da and da < 0.7,
    ))
    # ROA
    roa = f.get("Attr1", float("nan"))
    cards.append(RatioCard(
        "Return on Assets", _fmt(roa, pct=True),
        "Net profit per rupee of assets. Negative means the company is loss-making.",
        roa == roa and roa > 0,
    ))
    # Equity ratio
    eq = f.get("Attr10", float("nan"))
    cards.append(RatioCard(
        "Equity Ratio", _fmt(eq, pct=True),
        "Owners' capital as a share of assets. Negative signals eroded net worth.",
        eq == eq and eq > 0.2,
    ))
    return cards


# ------------------------------------------------------------------ stress test
def stress(name: str, op_shock_pct: float = 0.0, leverage_pct: float = 0.0,
           interest_bps: float = 0.0, inflation_pp: float = 0.0, gdp_pp: float = 0.0):
    """Recompute the Altman-anchored score under a macro + company-specific scenario.

    Delegates to the pure `src.scoring.stress` module (economic transmission lives there,
    unit-testable without Streamlit). Positional args stay backward compatible: the first
    two are the company-specific operating-profit and leverage shocks; the macro levers
    are keyword. All feed the underlying line items and recompute Altman once -- linear,
    so instant.
    """
    from foresight import Scenario, run

    rec, prior = _ROSTER[name]
    sc = Scenario(interest_bps=interest_bps, inflation_pp=inflation_pp, gdp_pp=gdp_pp,
                  op_shock_pct=op_shock_pct, leverage_pct=leverage_pct)
    return run(rec, _SECTOR.get(name, ""), sc, prior=prior)


# --------------------------------------------------------------- review economics
@st.cache_resource(show_spinner=False)
def _economics_artifact() -> dict:
    """The precomputed catch curve, from validation OOF (models/review_economics.json).

    Baked at build time so the app never loads the Polish training data at runtime; the
    curve is fixed, and every cost scenario is cheap arithmetic on top of it.
    """
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "models" / "review_economics.json"
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def economics(cost_per_miss_lakh: float, cost_per_review_lakh: float) -> dict:
    """Cost of every review budget, and the cost-minimising policy, for both screens.

    Ranking firms by risk and reviewing the top K, expected cost is
    `cost_per_miss * (distress still un-reviewed) + cost_per_review * K`. Returns the full
    curve plus the argmin for Foresight and for the Altman screen, so the tab can render
    the tradeoff live as the user moves the two cost sliders.
    """
    art = _economics_artifact()
    n, pos = art["n"], art["total_distress"]
    cm, cr = cost_per_miss_lakh * 1e5, cost_per_review_lakh * 1e5

    def series(caught_key: str) -> list[dict]:
        out = []
        for row in art["curve"]:
            caught, reviewed = row[caught_key], row["reviewed"]
            out.append({
                "budget_pct": row["budget_pct"],
                "reviewed": reviewed,
                "caught": caught,
                "catch_rate": caught / pos if pos else 0.0,
                "precision": caught / reviewed if reviewed else 0.0,
                "cost_cr": (cm * (pos - caught) + cr * reviewed) / 1e7,
            })
        return out

    fore, alt = series("caught_foresight"), series("caught_altman")
    return {
        "n": n,
        "total_distress": pos,
        "foresight": fore,
        "altman": alt,
        "opt_foresight": min(fore, key=lambda r: r["cost_cr"]),
        "opt_altman": min(alt, key=lambda r: r["cost_cr"]),
        "review_everything_cr": cr * n / 1e7,
        "review_nothing_cr": cm * pos / 1e7,
    }

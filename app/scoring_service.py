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

from src.scoring.combined import CombinedRisk, fuse
from src.serving import demo_companies as demo
from src.serving.financial_score import FinancialScore, risk_from_z, score_company
from src.serving.screener import ScreenerFinancials, compute_features
from src.signals.composite import DigitalPulse
from src.signals.demo_signals import DEFAULT_AS_OF, has_signals, pulse_as_of

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
    from src.signals.sentiment import default_scorer

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


# Sector labels -- single source of truth in demo_companies so the UI and the narrative
# pre-warm produce identical prompts (and therefore identical cache keys).
from src.serving.demo_companies import SECTORS as _SECTOR


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
    """Score every company for the surveillance table (Module 8), worst risk first."""
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


# -------------------------------------------------- narrative (M6) + advice (M7)
@st.cache_data(show_spinner=False)
def narrative(name: str) -> tuple[str, str]:
    """AI narrative + its source ('llm' or 'rule-based'). Never raises."""
    from src.narrative.summary import generate

    return generate(combined(name), financial(name), digital(name), _SECTOR.get(name, ""))


@st.cache_data(show_spinner=False)
def advice(name: str):
    """Rule-based recommendations grouped by audience (Module 7)."""
    from src.narrative.recommendations import recommend

    rec, prior = _ROSTER[name]
    return recommend(rec, financial(name), digital(name), prior)


@st.cache_data(show_spinner=False)
def report_pdf(name: str) -> bytes:
    """Two-page executive memo (Module 9). Cached so repeat clicks are instant."""
    from src.reporting.executive_report import build_report

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

    from src.signals.demo_signals import pulse_as_of as _pulse

    scorer = _sentiment_scorer()
    fin = financial(name).risk_score
    points = [("Jun 2025", _d(2025, 6, 30)), ("Sep 2025", _d(2025, 9, 30)),
              ("Dec 2025", _d(2025, 12, 31)), ("Mar 2026", _d(2026, 3, 31)),
              ("Jun 2026", _d(2026, 6, 30))]
    return [TimelinePoint(label=l, digital=_pulse(name, w, scorer).composite_score,
                          financial=fin) for l, w in points]


# --------------------------------------------------------------- financial ratios (M1)
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
    """Key financial ratios with one line of context each (Module 1)."""
    rec, prior = _ROSTER[name]
    f = compute_features(rec, prior=prior)

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


# ------------------------------------------------------------------ stress test (M5)
def stress(name: str, op_shock_pct: float = 0.0, leverage_pct: float = 0.0,
           interest_bps: float = 0.0, inflation_pp: float = 0.0, gdp_pp: float = 0.0):
    """Recompute the Altman-anchored score under a macro + company-specific scenario.

    Delegates to the pure `src.scoring.stress` module (economic transmission lives there,
    unit-testable without Streamlit). Positional args stay backward compatible: the first
    two are the company-specific operating-profit and leverage shocks; the macro levers
    are keyword. All feed the underlying line items and recompute Altman once -- linear,
    so instant.
    """
    from src.scoring.stress import Scenario, run

    rec, prior = _ROSTER[name]
    sc = Scenario(interest_bps=interest_bps, inflation_pp=inflation_pp, gdp_pp=gdp_pp,
                  op_shock_pct=op_shock_pct, leverage_pct=leverage_pct)
    return run(rec, _SECTOR.get(name, ""), sc, prior=prior)

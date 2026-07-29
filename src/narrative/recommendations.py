"""Module 7 -- Risk Mitigation Recommendations.

**Rule-based by design, never LLM-generated** (AGENTS.md M7). These sit next to a number
a credit committee may act on, so they must be reproducible and incapable of inventing a
fact. Every rule is tied to actual metric values, not generic advice.

Output is split by audience, which is what shows the platform serves multiple buyers:
  * FOR LENDERS / INVESTORS -- what to do if you have exposure.
  * FOR MANAGEMENT -- what to do if this is your own company.

Rules fire independently and are ranked by priority, so a company can trigger several.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from ..serving.financial_score import FinancialScore
from ..serving.screener import ScreenerFinancials, compute_features
from ..signals.base import SignalKind
from ..signals.composite import DigitalPulse


class Audience(str, Enum):
    LENDER = "For Lenders / Investors"
    MANAGEMENT = "For Management"


@dataclass(frozen=True)
class Recommendation:
    audience: Audience
    title: str
    action: str
    priority: int  # 1 = most urgent


@dataclass
class _Ctx:
    """Everything the rules read. Missing values are NaN/None and rules must tolerate it."""
    company: str
    fin: FinancialScore
    interest_cover: float
    #: total liabilities / assets -- INCLUDES trade payables and provisions, so this is
    #: not "debt". FMCG names run high here on supplier credit while carrying no borrowing.
    debt_to_assets: float
    #: interest-bearing borrowings / assets. Use THIS for any statement about leverage or
    #: debt, never `debt_to_assets`, or debt-free companies get told they are levered.
    borrowings_to_assets: float
    equity_ratio: float
    roa: float
    op_margin: float
    wc_to_assets: float
    digital: Optional[DigitalPulse]

    def sig(self, kind: SignalKind) -> Optional[float]:
        if self.digital is None:
            return None
        r = self.digital.by_kind(kind)
        return None if r is None else r.risk_score

    def sig_label(self, kind: SignalKind) -> str:
        if self.digital is None:
            return ""
        r = self.digital.by_kind(kind)
        return "" if r is None else r.label

    def sig_raw(self, kind: SignalKind) -> Optional[float]:
        if self.digital is None:
            return None
        r = self.digital.by_kind(kind)
        return None if r is None else r.raw


def _ok(x: Optional[float]) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


Rule = Callable[[_Ctx], Optional[Recommendation]]
_RULES: list[Rule] = []


def _rule(fn: Rule) -> Rule:
    _RULES.append(fn)
    return fn


# ------------------------------------------------------------------ lender rules
@_rule
def _r_negative_equity(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.equity_ratio) and c.equity_ratio < 0:
        return Recommendation(
            Audience.LENDER, "Negative net worth",
            f"Book equity is negative ({c.equity_ratio:.0%} of assets): liabilities exceed "
            "assets, so unsecured exposure has no equity cushion behind it. Escalate to "
            "special-mention, re-verify collateral values and enforce security where held.", 1)
    return None


@_rule
def _r_cannot_service_debt(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and c.interest_cover < 1.0:
        return Recommendation(
            Audience.LENDER, "Interest not covered by operations",
            f"Interest coverage is {c.interest_cover:.2f}x - operating profit does not cover "
            "the interest bill, so debt service is being funded from reserves or new "
            "borrowing. Treat further drawdown requests as refinancing, not growth.", 1)
    return None


@_rule
def _r_thin_cover_high_leverage(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and _ok(c.debt_to_assets) and 1.0 <= c.interest_cover < 2.0 and c.debt_to_assets > 0.65:
        return Recommendation(
            Audience.LENDER, "Debt servicing capacity is thin relative to leverage",
            f"Coverage of {c.interest_cover:.2f}x against liabilities at {c.debt_to_assets:.0%} "
            "of assets leaves little headroom for a rate or earnings shock. Reassess covenant "
            "levels and consider tightening reporting frequency to monthly.", 2)
    return None


@_rule
def _r_distress_zone(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.zone == "Distress":
        return Recommendation(
            Audience.LENDER, "Altman Z-Score in the distress zone",
            f"Z'' of {c.fin.z_score:.2f} sits below the 1.10 distress threshold. Move the "
            "name to the watchlist, refresh the internal rating, and require updated "
            "management accounts before any limit renewal.", 2)
    return None


@_rule
def _r_loss_making(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.roa) and c.roa < 0:
        return Recommendation(
            Audience.LENDER, "Loss-making at the asset level",
            f"Return on assets is {c.roa:.0%}: the asset base is destroying capital rather "
            "than generating it. Confirm whether losses are one-off or structural before "
            "extending the facility.", 2)
    return None


@_rule
def _r_negative_working_capital(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.wc_to_assets) and c.wc_to_assets < -0.10 and _ok(c.interest_cover) and c.interest_cover < 2.0:
        return Recommendation(
            Audience.LENDER, "Negative working capital alongside weak coverage",
            f"Working capital is {c.wc_to_assets:.0%} of assets while coverage is only "
            f"{c.interest_cover:.2f}x. Short-term obligations are being met from operating "
            "inflows with no buffer - a single collection delay could trigger default.", 2)
    return None


@_rule
def _r_leadership_exodus(c: _Ctx) -> Optional[Recommendation]:
    n = c.sig_raw(SignalKind.LEADERSHIP)
    if n is not None and n > 2:
        return Recommendation(
            Audience.LENDER, "Senior leadership turnover",
            f"{int(n)} senior departures filed in the last six months. Concentrated exits at "
            "board or KMP level often precede disclosure events; request a governance update "
            "and confirm who now holds financial authority.", 2)
    return None


@_rule
def _r_operational_stress(c: _Ctx) -> Optional[Recommendation]:
    hire = c.sig_label(SignalKind.HIRING)
    sent = c.sig(SignalKind.NEWS_SENTIMENT)
    if hire == "Contracting" and sent is not None and sent >= 50:
        return Recommendation(
            Audience.LENDER, "Workforce contraction with negative coverage",
            "Hiring is contracting while news sentiment is negative - a pattern that "
            "indicates operational stress not yet reflected in the reported financials. "
            "Request updated management accounts rather than waiting for the next filing.", 2)
    return None


@_rule
def _r_digital_worse_than_financial(c: _Ctx) -> Optional[Recommendation]:
    if c.digital is None:
        return None
    gap = c.digital.composite_score - c.fin.risk_score
    if gap > 15:
        return Recommendation(
            Audience.LENDER, "Market signals weaker than the financials suggest",
            f"Digital signals score {c.digital.composite_score:.0f} against a financial score "
            f"of {c.fin.risk_score:.0f}. Where market signals lead the statements, the next "
            "reporting period often confirms them - bring forward the review date.", 3)
    return None


@_rule
def _r_no_digital_coverage(c: _Ctx) -> Optional[Recommendation]:
    if c.digital is None:
        return Recommendation(
            Audience.LENDER, "No market-intelligence coverage",
            "This assessment rests on reported financials alone. Establish news, hiring and "
            "filing monitoring for this name so deterioration between reporting dates is "
            "visible.", 4)
    return None


@_rule
def _r_grey_zone_watch(c: _Ctx) -> Optional[Recommendation]:
    """The watchlist band is where a lender most needs guidance -- without this, a company
    that is neither clearly safe nor clearly distressed produces no lender action."""
    if c.fin.zone == "Grey" or c.fin.band == "Watch":
        lev = (f" with liabilities at {c.debt_to_assets:.0%} of assets"
               if _ok(c.debt_to_assets) and c.debt_to_assets > 0.6 else "")
        return Recommendation(
            Audience.LENDER, "Grey zone - neither clearly safe nor distressed",
            f"Z'' of {c.fin.z_score:.2f} falls between the distress and safe thresholds{lev}. "
            "This is the band where outcomes diverge most: move to semi-annual review, "
            "confirm covenant headroom, and track the market signals for early deterioration.", 3)
    return None


@_rule
def _r_leverage_concentration(c: _Ctx) -> Optional[Recommendation]:
    # Gated on actual borrowings: a company funded by supplier credit is not "levered".
    if _ok(c.borrowings_to_assets) and c.borrowings_to_assets >= 0.30:
        return Recommendation(
            Audience.LENDER, "High balance-sheet leverage",
            f"Interest-bearing borrowings fund {c.borrowings_to_assets:.0%} of the asset base, "
            "so a modest fall in asset values would erode the equity cushion. Re-test security "
            "cover against current, not historic, valuations.", 3)
    return None


@_rule
def _r_healthy_but_levered(c: _Ctx) -> Optional[Recommendation]:
    if (c.fin.band == "Healthy" and _ok(c.borrowings_to_assets) and c.borrowings_to_assets > 0.20
            and _ok(c.interest_cover) and c.interest_cover >= 2.0):
        return Recommendation(
            Audience.LENDER, "Sound today, but leverage is material",
            f"Borrowings are {c.borrowings_to_assets:.0%} of assets even though coverage is "
            f"comfortable at {c.interest_cover:.1f}x. Monitor covenant headroom under a rate "
            "rise; the profile is sound but not shock-proof.", 4)
    return None


@_rule
def _r_healthy_standard_cycle(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.band == "Healthy" and (c.digital is None or c.digital.band == "Healthy"):
        return Recommendation(
            Audience.LENDER, "No action indicated",
            f"Financial and market signals are both in the healthy band (Z'' "
            f"{c.fin.z_score:.1f}). Maintain the standard annual review cycle; no additional "
            "monitoring is warranted on current evidence.", 5)
    return None


# -------------------------------------------------------------- management rules
@_rule
def _r_mgmt_recapitalize(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.equity_ratio) and c.equity_ratio < 0:
        return Recommendation(
            Audience.MANAGEMENT, "Restore the equity base",
            "Net worth is negative, which constrains new borrowing and may breach covenants. "
            "Priority is a recapitalisation - rights issue, promoter infusion, or converting "
            "debt to equity - before refinancing terms deteriorate further.", 1)
    return None


@_rule
def _r_mgmt_refinance(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and c.interest_cover < 1.5:
        return Recommendation(
            Audience.MANAGEMENT, "Reduce the debt service burden",
            f"At {c.interest_cover:.2f}x coverage, interest is consuming most or all of "
            "operating profit. Open refinancing discussions early - extending maturities and "
            "resetting covenants is far cheaper while the company is still performing.", 1)
    return None


@_rule
def _r_mgmt_working_capital(c: _Ctx) -> Optional[Recommendation]:
    # Negative working capital is a *strength* in FMCG and retail -- suppliers fund the
    # business. It is only a problem when the company is not otherwise healthy, so this
    # rule is gated on the band. Telling a debt-free FMCG name to "close the gap" would be
    # wrong advice and would cost credibility instantly.
    if c.fin.band == "Healthy":
        return None
    if _ok(c.wc_to_assets) and c.wc_to_assets < -0.10:
        return Recommendation(
            Audience.MANAGEMENT, "Close the working capital gap",
            f"Working capital is {c.wc_to_assets:.0%} of assets. Negotiate a committed "
            "working-capital facility and tighten receivables collection so operations are "
            "not dependent on continuous supplier financing.", 2)
    return None


@_rule
def _r_mgmt_margins(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.op_margin) and c.op_margin < 0.05:
        return Recommendation(
            Audience.MANAGEMENT, "Rebuild operating margin",
            f"Operating margin is {c.op_margin:.1%}, leaving no absorption for input-cost or "
            "demand shocks. A structural cost review will do more for the risk profile than "
            "further balance-sheet engineering.", 2)
    return None


@_rule
def _r_mgmt_leadership(c: _Ctx) -> Optional[Recommendation]:
    n = c.sig_raw(SignalKind.LEADERSHIP)
    if n is not None and n >= 2:
        return Recommendation(
            Audience.MANAGEMENT, "Stabilise the leadership team",
            f"{int(n)} senior departures in six months is visible to lenders and rating "
            "agencies. Publish a clear succession and retention plan; unexplained churn is "
            "read as a governance signal regardless of the underlying reason.", 2)
    return None


@_rule
def _r_mgmt_workforce(c: _Ctx) -> Optional[Recommendation]:
    if c.sig_label(SignalKind.HIRING) == "Contracting" or c.sig_label(SignalKind.EMPLOYEE) == "Declining":
        return Recommendation(
            Audience.MANAGEMENT, "Address workforce signals",
            "Hiring contraction and falling employee sentiment are externally visible and "
            "feed third-party risk models. Communicate the operating plan internally before "
            "attrition compounds the problem.", 3)
    return None


@_rule
def _r_mgmt_communication(c: _Ctx) -> Optional[Recommendation]:
    s = c.sig(SignalKind.NEWS_SENTIMENT)
    if s is not None and s >= 55:
        return Recommendation(
            Audience.MANAGEMENT, "Get ahead of the narrative",
            "Recent coverage is materially negative. Proactive disclosure on liquidity and "
            "the funding plan is more effective than silence - lenders price uncertainty "
            "more harshly than bad news.", 3)
    return None


@_rule
def _r_mgmt_maintain(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.band == "Healthy":
        return Recommendation(
            Audience.MANAGEMENT, "Preserve the current position",
            f"The balance sheet is sound (Z'' {c.fin.z_score:.1f}). Maintain the current "
            "leverage discipline; the main risk to this profile is debt-funded expansion "
            "rather than trading performance.", 5)
    return None


# -------------------------------------------------------------------- public API
def build_context(
    rec: ScreenerFinancials,
    fin: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    prior: Optional[ScreenerFinancials] = None,
) -> _Ctx:
    f = compute_features(rec, prior=prior)
    bta = (rec.borrowings / rec.total_assets) if rec.total_assets else float("nan")
    return _Ctx(
        company=rec.company, fin=fin,
        interest_cover=f.get("Attr27", float("nan")),
        debt_to_assets=f.get("Attr2", float("nan")),
        borrowings_to_assets=bta,
        equity_ratio=f.get("Attr10", float("nan")),
        roa=f.get("Attr1", float("nan")),
        op_margin=f.get("Attr42", float("nan")),
        wc_to_assets=f.get("Attr3", float("nan")),
        digital=digital,
    )


def recommend(
    rec: ScreenerFinancials,
    fin: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    prior: Optional[ScreenerFinancials] = None,
) -> dict[Audience, list[Recommendation]]:
    """Fire every rule and group the hits by audience, most urgent first."""
    ctx = build_context(rec, fin, digital, prior)
    out: dict[Audience, list[Recommendation]] = {Audience.LENDER: [], Audience.MANAGEMENT: []}
    for rule in _RULES:
        r = rule(ctx)
        if r is not None:
            out[r.audience].append(r)
    for aud in out:
        out[aud].sort(key=lambda r: r.priority)
    return out


def rule_count() -> int:
    return len(_RULES)

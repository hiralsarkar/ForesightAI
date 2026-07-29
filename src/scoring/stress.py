"""Module 5 - Macroeconomic Stress Testing.

Maps macro shocks to a company's Altman-anchored risk score through explicit, defensible
economic channels. Three macro levers (the ones the problem statement names) plus two
company-specific levers. USD/INR and sector credit spread are deliberately excluded:
USD/INR helps exporters and hurts importers, so a single slider would move IT firms the
wrong way; credit spread just duplicates the interest-rate channel.

**Transmission (each shock feeds a line item; Altman is recomputed once at the end):**

* Interest rate (Δbps, sustained `horizon` years). Interest sits *below* EBIT, so a rate
  shock does not touch WC/TA or EBIT/TA. It raises interest expense - the headline credit
  channel, shown as the change in **interest coverage** - and, sustained over the horizon,
  erodes retained earnings (equity). Modelled on the floating-rate share of borrowings.
* Inflation (Δpp). Compresses the operating margin by a sector-specific pass-through
  factor: Δoperating_profit = −inflation_beta x Δpp x sales. Flows to EBIT/TA and reserves.
* GDP growth (Δpp). A demand shock: Δsales = gdp_beta x Δpp x sales, of which a
  sector-specific incremental margin drops to operating profit.

**Sector elasticities are explicit and rationale'd** - a single global number is
indefensible (an airline, a commodity producer and an IT firm have wildly different
pricing power and operating leverage). Defaults are conservative.

Everything is a linear recompute, so the panel updates instantly (<2s trivially).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..features.altman import altman_z
from ..serving.financial_score import risk_from_z
from ..serving.screener import ScreenerFinancials, compute_features


@dataclass(frozen=True)
class SectorProfile:
    """Macro sensitivities for a sector, each with a one-line rationale."""

    inflation_beta: float   # operating-margin compression per 1pp inflation (frac of sales)
    gdp_beta: float         # sales change per 1pp GDP (%)
    incremental_margin: float  # frac of a sales change that reaches operating profit
    floating_share: float   # share of borrowings that reprices with rates
    rationale: str


# Conservative, sector-differentiated defaults.
_SECTORS: dict[str, SectorProfile] = {
    "Airline": SectorProfile(
        0.010, 1.6, 0.45, 0.70,
        "Thin pricing power and fuel-cost exposure (high inflation hit); highly cyclical, "
        "high fixed cost (demand and operating-leverage sensitive)."),
    "Electric Vehicles": SectorProfile(
        0.010, 1.5, 0.40, 0.70,
        "Imported inputs and price-competitive (high inflation hit); discretionary demand."),
    "Telecom": SectorProfile(
        0.005, 0.5, 0.55, 0.70,
        "Some subscription pricing power (moderate inflation hit); essential service so "
        "demand is defensive, but very high fixed cost amplifies any revenue change."),
    "Metals & Mining": SectorProfile(
        0.003, 1.5, 0.45, 0.60,
        "Commodity producer - output prices broadly track inflation, so input and output "
        "inflation largely offset (low net margin hit); highly cyclical demand."),
    "IT Services": SectorProfile(
        0.003, 0.9, 0.30, 0.50,
        "Strong contractual pricing power and repriceable labour (low inflation hit); "
        "largely variable cost base and moderate cyclicality."),
    "Fintech": SectorProfile(
        0.005, 1.0, 0.35, 0.50,
        "Moderate pricing power and moderate demand cyclicality."),
}
_DEFAULT = SectorProfile(0.006, 1.0, 0.40, 0.60, "Generic non-financial corporate defaults.")


def sector_profile(sector: str) -> SectorProfile:
    return _SECTORS.get(sector, _DEFAULT)


@dataclass(frozen=True)
class Scenario:
    """A stress scenario. All fields default to 'no change'."""

    interest_bps: float = 0.0     # e.g. +200 = rates rise 2 percentage points
    inflation_pp: float = 0.0     # e.g. +3 = inflation 3pp higher
    gdp_pp: float = 0.0           # e.g. -2 = GDP growth 2pp lower
    op_shock_pct: float = 0.0     # company-specific direct hit to operating profit (%)
    leverage_pct: float = 0.0     # company-specific change in borrowings (%)
    horizon_years: int = 3        # interest-rate shocks are sustained this long


@dataclass
class StressResult:
    base_score: float
    stressed_score: float
    base_z: float
    stressed_z: float
    base_coverage: float          # operating profit / interest, before
    stressed_coverage: float      # after
    contributions: dict           # channel -> score delta it alone would cause

    @property
    def delta(self) -> float:
        return round(self.stressed_score - self.base_score, 1)


def _coverage(rec: ScreenerFinancials, extra_interest: float = 0.0) -> float:
    interest = rec.interest + extra_interest
    return float("nan") if interest == 0 else rec.operating_profit / interest


def apply_to_financials(rec: ScreenerFinancials, sc: Scenario, prof: SectorProfile) -> tuple[ScreenerFinancials, float]:
    """Return the shocked financials and the incremental annual interest cost.

    All channels compose on the underlying line items so a combined scenario interacts
    correctly and never double-counts.
    """
    op = rec.operating_profit
    reserves = rec.reserves

    # --- Interest rate: extra interest, sustained over the horizon, erodes equity. ---
    extra_interest = rec.borrowings * prof.floating_share * (sc.interest_bps / 10000.0)
    reserves -= extra_interest * sc.horizon_years

    # --- Inflation: margin compression. ---
    op += -prof.inflation_beta * sc.inflation_pp * rec.sales

    # --- GDP: demand shock through sales -> operating profit at the incremental margin. ---
    d_sales = prof.gdp_beta * (sc.gdp_pp / 100.0) * rec.sales
    op += d_sales * prof.incremental_margin

    # --- Company-specific direct levers. ---
    op += rec.operating_profit * (sc.op_shock_pct / 100.0)
    debt_delta = rec.borrowings * (sc.leverage_pct / 100.0)
    reserves -= debt_delta

    # Operating-profit changes flow to retained earnings for the year.
    reserves += (op - rec.operating_profit)

    shocked = replace(
        rec,
        operating_profit=op,
        profit_before_tax=rec.profit_before_tax + (op - rec.operating_profit) - extra_interest,
        reserves=reserves,
        borrowings=rec.borrowings + debt_delta,
    )
    return shocked, extra_interest


def _score(rec: ScreenerFinancials, prior) -> float:
    import pandas as pd
    z = float(altman_z(pd.DataFrame([compute_features(rec, prior=prior)]), "z2").iloc[0])
    return z


def run(rec: ScreenerFinancials, sector: str, sc: Scenario, prior=None) -> StressResult:
    """Apply a scenario and return the full before/after picture."""
    prof = sector_profile(sector)

    base_z = _score(rec, prior)
    shocked, extra_int = apply_to_financials(rec, sc, prof)
    stressed_z = _score(shocked, prior)

    # Per-channel attribution: what each lever alone would do to the score. Lets the UI
    # say "rising rates account for N of the M-point move".
    channels = {
        "Interest rate": Scenario(interest_bps=sc.interest_bps, horizon_years=sc.horizon_years),
        "Inflation": Scenario(inflation_pp=sc.inflation_pp),
        "GDP growth": Scenario(gdp_pp=sc.gdp_pp),
        "Company-specific": Scenario(op_shock_pct=sc.op_shock_pct, leverage_pct=sc.leverage_pct),
    }
    base_score = risk_from_z(base_z)
    contributions = {}
    for name, one in channels.items():
        one_shocked, _ = apply_to_financials(rec, one, prof)
        contributions[name] = round(risk_from_z(_score(one_shocked, prior)) - base_score, 1)

    # Coverage after = shocked operating profit / shocked interest.
    new_interest = rec.interest + extra_int
    stressed_cov = (shocked.operating_profit / new_interest) if new_interest else float("nan")

    return StressResult(
        base_score=round(base_score, 1),
        stressed_score=round(risk_from_z(stressed_z), 1),
        base_z=round(base_z, 2),
        stressed_z=round(stressed_z, 2),
        base_coverage=round(_coverage(rec), 2),
        stressed_coverage=round(stressed_cov, 2),
        contributions=contributions,
    )

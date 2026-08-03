"""Serving bridge: Screener.in financials -> Polish `AttrN` feature vector.

This is the linchpin of the whole product. The model trains on Polish `Attr1..Attr64`;
this module is what lets it score an *Indian* company. Without it there is no demo.

**What Screener actually gives us (verified against live pages, July 2026).** The
default statements are heavily aggregated -- there is no inventory, receivables, or
current-assets line. So the current-ratio family is not computable from the balance
sheet and comes back NaN. The model routes NaN natively by design, so this
is graceful, but see the structural-missingness warning below.

    P&L         Sales, Expenses, Operating Profit, Other Income, Interest,
                Depreciation, Profit before tax, Net Profit
    Balance     Equity Capital, Reserves, Borrowings, Other Liabilities,
                Total Liabilities (= balance-sheet total), Fixed Assets, CWIP,
                Investments, Other Assets, Total Assets
    Ratios      Debtor Days, Inventory Days, Days Payable, Cash Conversion Cycle,
                Working Capital Days   (a separate section, per-year)

Derived intermediates:
    equity   = Equity Capital + Reserves            (Reserves ~= retained earnings)
    debt     = Borrowings
    ebit     = Profit before tax + Interest
    ebitda   = ebit + Depreciation
    tot_liab = Total Assets - equity                 (all non-equity claims)
    working_capital = Working Capital Days * Sales / 365   (recovers WC without the
                      current-asset breakdown)

**⚠️ Structural vs informative missingness.**
In training, a missing feature was *informative* -- distressed Polish firms failed to
report. Here, the current-ratio family is missing *structurally* -- Screener never
breaks it out, for healthy and distressed companies alike. If the model learned
"feature missing -> distress" and every Indian firm is missing the same features, every
Indian score gets a systematic upward nudge. This module cannot fix that; it only
surfaces it. The gate is `validate_healthy_controls()` downstream: healthy companies
must land in the healthy band, or the structural-NaN features must be neutralised on the
serving path. Do not trust any Indian score until that passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from ..features.polish_schema import N_ATTRS, serving_feature_set


class Provenance(str, Enum):
    """Where a served feature's value comes from, for coverage auditing."""

    STATEMENT = "statement"      # from the aggregated P&L / balance sheet
    DAYS_RATIO = "days_ratio"    # from Screener's separate ratios section
    STRUCTURAL_NA = "structural" # not computable from Screener's aggregation -> NaN


@dataclass
class ScreenerFinancials:
    """One company-year of Screener data. All amounts in the same currency unit (Cr).

    Point-in-time: the values must be as reported *for that financial year*, not
    restated later. `year` is the March-ending fiscal year.
    """

    company: str
    year: int

    # P&L
    sales: float
    expenses: float
    operating_profit: float
    other_income: float
    interest: float
    depreciation: float
    profit_before_tax: float
    net_profit: float

    # Balance sheet
    equity_capital: float
    reserves: float
    borrowings: float
    other_liabilities: float
    total_assets: float
    fixed_assets: float
    investments: float = 0.0
    cwip: float = 0.0
    other_assets: float = 0.0

    # Ratios section (optional; None -> the dependent AttrN become NaN)
    debtor_days: Optional[float] = None
    inventory_days: Optional[float] = None
    days_payable: Optional[float] = None
    working_capital_days: Optional[float] = None

    ticker: str = ""

    # ---- derived intermediates -------------------------------------------------
    @property
    def equity(self) -> float:
        return self.equity_capital + self.reserves

    @property
    def total_liabilities(self) -> float:
        """All non-equity claims. Screener's own 'Total Liabilities' row is the
        balance-sheet total (== total assets), which is *not* what the Polish
        `total liabilities` attributes mean."""
        return self.total_assets - self.equity

    @property
    def ebit(self) -> float:
        return self.profit_before_tax + self.interest

    @property
    def ebitda(self) -> float:
        return self.ebit + self.depreciation

    @property
    def working_capital(self) -> Optional[float]:
        if self.working_capital_days is None:
            return None
        return self.working_capital_days * self.sales / 365.0


def _safe(numer: float, denom: float) -> float:
    """Ratio with a NaN guard. A zero denominator is undefined, not zero or infinity --
    returning inf would poison the feature and any downstream aggregate."""
    if denom is None or denom == 0 or (isinstance(denom, float) and math.isnan(denom)):
        return float("nan")
    if numer is None or (isinstance(numer, float) and math.isnan(numer)):
        return float("nan")
    return numer / denom


def compute_features(
    fin: ScreenerFinancials,
    prior: Optional[ScreenerFinancials] = None,
    prior3: Optional[list[ScreenerFinancials]] = None,
) -> dict[str, float]:
    """Map one company-year to the serving `AttrN` features.

    Returns a dict over exactly `serving_feature_set()`; features Screener cannot
    support are NaN. Formulas follow `polish_schema` definitions, translated into
    Screener quantities.

    `prior` is the immediately preceding fiscal year, needed for growth features
    (Attr21). `prior3` is up to three preceding years, for the 3-year gross-profit
    feature (Attr24). Both optional -- absent history just yields NaN for those, which
    the model routes natively.
    """
    ta = fin.total_assets
    eq = fin.equity
    tl = fin.total_liabilities
    sales = fin.sales
    wc = fin.working_capital
    op = fin.operating_profit

    # Every serving attribute, computed where Screener allows and NaN otherwise.
    # Grouped by the intermediate they depend on so the provenance is auditable.
    v: dict[str, float] = {
        # --- from the aggregated statements -----------------------------------
        "Attr1": _safe(fin.net_profit, ta),                       # ROA
        "Attr2": _safe(tl, ta),                                   # debt/assets
        "Attr6": _safe(fin.reserves, ta),                         # retained earnings/TA
        "Attr7": _safe(fin.ebit, ta),                             # EBIT/TA
        "Attr8": _safe(eq, tl),                                   # equity/liabilities
        "Attr9": _safe(sales, ta),                                # asset turnover
        "Attr10": _safe(eq, ta),                                  # equity ratio
        "Attr17": _safe(ta, tl),                                  # asset cover of debt
        "Attr22": _safe(fin.operating_profit, ta),               # op profit/TA
        "Attr23": _safe(fin.net_profit, sales),                  # net margin
        "Attr25": _safe(fin.reserves, ta),                       # reserves/TA
        "Attr26": _safe(fin.net_profit + fin.depreciation, tl),  # cash earnings/debt
        "Attr27": _safe(fin.operating_profit, fin.interest),     # interest coverage
        "Attr29": math.log(ta) if ta and ta > 0 else float("nan"),  # log size
        "Attr36": _safe(sales, ta),                              # total asset turnover
        "Attr42": _safe(fin.operating_profit, sales),           # operating margin
        "Attr48": _safe(fin.ebitda, ta),                         # EBITDA/TA
        "Attr49": _safe(fin.ebitda, sales),                     # EBITDA margin
        "Attr53": _safe(eq, fin.fixed_assets),                  # equity/fixed assets
        "Attr58": _safe(sales - fin.operating_profit, sales),   # cost/sales
        "Attr64": _safe(sales, fin.fixed_assets),               # fixed asset turnover
        "Attr34": _safe(fin.expenses, tl),                      # opex / total liabilities
        "Attr13": _safe(op + fin.depreciation, sales),          # cash gross margin
        "Attr41": _safe(tl, (op + fin.depreciation) * (12.0 / 365.0)),  # debt/annualised op CF
        "Attr30": _safe(tl, sales),                             # ~net debt/sales (no cash split)
        "Attr16": _safe(fin.net_profit + fin.depreciation, tl), # cash flow to debt (~Attr26)
        "Attr24": float("nan"),  # filled from prior3 below
        "Attr21": float("nan"),  # filled from prior below
        # gross-profit proxies: Screener has no COGS split, so gross ~= operating.
        "Attr11": _safe(fin.ebit, ta),                          # pre-financing RoA (~EBIT/TA)
        "Attr14": _safe(fin.ebit, ta),                          # pre-interest RoA
        "Attr18": _safe(fin.operating_profit, ta),             # gross RoA (~op/TA)
        "Attr19": _safe(fin.operating_profit, sales),          # gross margin (~OPM)
        "Attr31": _safe(fin.ebit, sales),                      # pre-interest margin
        "Attr35": _safe(fin.operating_profit, ta),             # sales profit/TA
        "Attr39": _safe(fin.operating_profit, sales),          # sales profit margin
        "Attr56": _safe(fin.operating_profit, sales),          # gross margin on cost (~OPM)
        # --- from Working Capital Days ----------------------------------------
        "Attr3": _safe(wc, ta) if wc is not None else float("nan"),   # WC/TA
        "Attr28": _safe(wc, fin.fixed_assets) if wc is not None else float("nan"),
        # --- from the days-ratio section --------------------------------------
        "Attr20": fin.inventory_days if fin.inventory_days is not None else float("nan"),
        "Attr44": fin.debtor_days if fin.debtor_days is not None else float("nan"),
        "Attr47": fin.inventory_days if fin.inventory_days is not None else float("nan"),
        "Attr32": fin.days_payable if fin.days_payable is not None else float("nan"),
        "Attr62": fin.days_payable if fin.days_payable is not None else float("nan"),
        "Attr43": (
            (fin.debtor_days + fin.inventory_days)
            if (fin.debtor_days is not None and fin.inventory_days is not None)
            else float("nan")
        ),
        "Attr60": _safe(365.0, fin.inventory_days) if fin.inventory_days else float("nan"),
        "Attr61": _safe(365.0, fin.debtor_days) if fin.debtor_days else float("nan"),
    }

    # Growth feature (Attr21 = sales_n / sales_{n-1}), from immediate prior year.
    if prior is not None and prior.sales:
        v["Attr21"] = _safe(sales, prior.sales)

    # 3-year gross-profit / total assets (Attr24): cumulative operating profit over the
    # last three years divided by current total assets. Uses whatever history exists.
    if prior3:
        window = [fin] + list(prior3)
        gp3 = sum(r.operating_profit for r in window[:3])
        if len(window) >= 2:  # require at least two years to be meaningful
            v["Attr24"] = _safe(gp3, ta)

    # Everything else in the serving set is structurally unavailable from Screener
    # (current-ratio family, and long/short-term debt split) -> NaN.
    out: dict[str, float] = {}
    for attr in serving_feature_set(include_derived=True):
        out[attr] = float(v.get(attr, float("nan")))
    return out


def compute_with_history(
    history: list[ScreenerFinancials], year: int
) -> dict[str, float]:
    """Compute features for `year`, wiring prior-year context from `history`.

    `history` is a company's records in any order; growth and 3-year features are
    filled from the years preceding `year` where present.
    """
    by_year = {r.year: r for r in history}
    if year not in by_year:
        raise KeyError(f"no record for {year}; have {sorted(by_year)}")
    prior = by_year.get(year - 1)
    prior3 = [by_year[y] for y in (year - 1, year - 2, year - 3) if y in by_year]
    return compute_features(by_year[year], prior=prior, prior3=prior3 or None)


def to_feature_row(
    fin: ScreenerFinancials,
    features: list[str],
    prior: Optional[ScreenerFinancials] = None,
    prior3: Optional[list[ScreenerFinancials]] = None,
) -> np.ndarray:
    """Feature vector aligned to a model's `features` order, NaN-filled off-serving."""
    computed = compute_features(fin, prior=prior, prior3=prior3)
    return np.array([computed.get(f, float("nan")) for f in features], dtype="float64")


def screener_feature_set() -> tuple[str, ...]:
    """The features the bridge can populate given complete inputs + history.

    **This is the true serving feature set** and what the served model must train on.
    Derived empirically from a fully-populated synthetic
    record so it can never drift from what `compute_features` actually assigns: if a
    formula is added or removed above, this set tracks it automatically.
    """
    probe = ScreenerFinancials(
        company="_probe", year=2020,
        sales=1000, expenses=800, operating_profit=200, other_income=20, interest=10,
        depreciation=30, profit_before_tax=210, net_profit=150,
        equity_capital=50, reserves=500, borrowings=200, other_liabilities=250,
        total_assets=1000, fixed_assets=400, investments=100, cwip=20, other_assets=80,
        debtor_days=50, inventory_days=40, days_payable=45, working_capital_days=30,
    )
    prior = ScreenerFinancials(
        company="_probe", year=2019,
        sales=900, expenses=720, operating_profit=180, other_income=18, interest=9,
        depreciation=28, profit_before_tax=189, net_profit=135,
        equity_capital=50, reserves=430, borrowings=210, other_liabilities=240,
        total_assets=930, fixed_assets=390,
    )
    computed = compute_features(probe, prior=prior, prior3=[prior])
    return tuple(a for a in serving_feature_set(include_derived=True)
                 if not math.isnan(computed[a]))


def coverage(fin: ScreenerFinancials) -> dict[str, float]:
    """How much of the serving feature set this company-year actually populates."""
    computed = compute_features(fin)
    serving = serving_feature_set(include_derived=True)
    present = sum(1 for a in serving if not math.isnan(computed[a]))
    return {
        "company": fin.company,
        "year": fin.year,
        "serving_features": len(serving),
        "populated": present,
        "structural_na": len(serving) - present,
        "coverage_pct": round(100 * present / len(serving), 1),
    }


def frame_from_records(records: list[ScreenerFinancials]) -> pd.DataFrame:
    """Build a tidy feature frame (one row per company-year) for batch scoring."""
    rows = []
    for fin in records:
        feats = compute_features(fin)
        rows.append({"company": fin.company, "ticker": fin.ticker, "year": fin.year, **feats})
    return pd.DataFrame(rows)

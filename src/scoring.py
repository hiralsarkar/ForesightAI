from __future__ import annotations

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


import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.features import N_ATTRS, serving_feature_set


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

"""Serving Financial Health Score, anchored on Altman Z''.

The serving work established that the GBM does not transfer to Indian companies while the
linear Altman Z'' does (see `docs/serving_indian_companies.md`). So the score a user sees for an Indian
company is derived from Z'', not from the GBM's calibrated probability.

Two things this module must get right:

1. **A 0-100 risk score monotonic in Z'', anchored on the published zone thresholds** so
   the bands line up exactly with Altman's own Distress / Grey / Safe boundaries. Higher
   score = higher risk (Z'' runs the other way: higher Z'' = safer).

2. **An exact 4-term decomposition** for the waterfall. Because Z'' is linear, each term
   is weight x value and the four terms *sum to the score* with no approximation -- a
   cleaner serving explanation than SHAP on a number we do not display.

The logistic anchoring solves for two fixed points:
    Z'' = 2.60 (safe threshold)     -> risk 25  (Healthy/Watch boundary)
    Z'' = 1.10 (distress threshold) -> risk 50  (Watch/Elevated boundary)
giving center z0 = 1.10 and steepness k = ln(3)/1.5 = 0.732. Everything below the
distress line lands >= 50 (Elevated/Critical); everything above the safe line lands < 25
(Healthy). This makes the gauge bands and the Altman zones the same statement.
"""


import math
from dataclasses import dataclass

import numpy as np

from src.features import Z_DOUBLE_PRIME, ZSpec, get_spec
from src.features import label_for

# Logistic anchor constants (derived above; see module docstring).
_Z0 = 1.10
_K = math.log(3.0) / (Z_DOUBLE_PRIME.safe_above - Z_DOUBLE_PRIME.distress_below)  # 0.732

# Human labels for the four Z'' terms, in the language the waterfall shows.
_TERM_LABELS: dict[str, str] = {
    "Attr3": "Working Capital / Total Assets",
    "Attr6": "Retained Earnings / Total Assets",
    "Attr7": "Operating Return on Assets (EBIT/TA)",
    "Attr8": "Book Equity / Total Liabilities",
}


def risk_from_z(z: float) -> float:
    """Map an Altman Z'' to a 0-100 risk score (higher = riskier)."""
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return float("nan")
    return float(100.0 / (1.0 + math.exp(_K * (z - _Z0))))


def z_from_risk(risk: float) -> float:
    """Inverse map -- useful for slider bounds and reporting a target Z''."""
    risk = min(max(risk, 1e-6), 100 - 1e-6)
    return _Z0 + math.log(100.0 / risk - 1.0) / _K


@dataclass(frozen=True)
class ZTerm:
    """One term of the Z'' decomposition."""

    attr: str
    label: str
    coefficient: float
    value: float
    contribution: float  # coefficient * value; exact, sums to Z''

    @property
    def direction(self) -> str:
        # Every Z'' term is risk-*decreasing* in value (higher = safer), so a negative
        # contribution is what pushes toward distress.
        return "reduces risk" if self.contribution > 0 else "increases risk"


@dataclass
class FinancialScore:
    company: str
    year: int
    z_score: float
    risk_score: float          # 0-100
    band: str
    zone: str                  # Altman zone label
    terms: list[ZTerm]
    missing_terms: list[str]   # Z'' components that were unavailable (filled 0)

    @property
    def is_reliable(self) -> bool:
        """A score built with a missing high-weight term (WC/TA at 6.56) is softer."""
        return not self.missing_terms

    def top_drivers(self, n: int = 3) -> list[ZTerm]:
        """Terms pushing hardest toward distress, worst first."""
        return sorted(self.terms, key=lambda t: t.contribution)[:n]


# Bands must match calibrate.BANDS so financial and combined scores speak one language.
_BANDS: tuple[tuple[float, str], ...] = (
    (25.0, "Healthy"),
    (50.0, "Watch"),
    (75.0, "Elevated Risk"),
    (100.1, "Critical"),
)


def band_for(risk: float) -> str:
    if math.isnan(risk):
        return "Unknown"
    for upper, name in _BANDS:
        if risk < upper:
            return name
    return _BANDS[-1][1]


def score_company(
    fin: ScreenerFinancials,
    prior: ScreenerFinancials | None = None,
    spec: ZSpec = Z_DOUBLE_PRIME,
    fill_missing: bool = True,
) -> FinancialScore:
    """Compute the Altman-anchored financial score and its exact decomposition.

    `fill_missing=True` substitutes 0 for an unavailable Z'' component (a neutral
    assumption -- e.g. missing working capital -> WC/TA = 0) so the score still computes,
    and records it in `missing_terms`. With `fill_missing=False`, any missing component
    yields a NaN score, matching the strict training-path Altman behaviour.
    """
    from src.features import zone as altman_zone

    feats = compute_features(fin, prior=prior)

    terms: list[ZTerm] = []
    missing: list[str] = []
    z = 0.0
    for attr, coef in spec.coefficients.items():
        val = feats.get(attr, float("nan"))
        if val is None or math.isnan(val):
            missing.append(_TERM_LABELS.get(attr, label_for(attr)))
            if not fill_missing:
                z = float("nan")
            val = 0.0
        contribution = coef * val
        if not math.isnan(z):
            z += contribution
        terms.append(
            ZTerm(attr, _TERM_LABELS.get(attr, label_for(attr)), coef, val, contribution)
        )

    if missing and not fill_missing:
        z = float("nan")

    risk = risk_from_z(z)
    import pandas as pd

    zone_label = (
        altman_zone(pd.Series([z]), "z2").iloc[0] if not math.isnan(z) else "Unknown"
    )

    return FinancialScore(
        company=fin.company,
        year=fin.year,
        z_score=z,
        risk_score=risk,
        band=band_for(risk),
        zone=str(zone_label),
        terms=terms,
        missing_terms=missing,
    )


def narrative(score: FinancialScore) -> str:
    """One-sentence plain-English summary of the financial score's drivers."""
    if math.isnan(score.risk_score):
        return f"{score.company} could not be scored: core balance-sheet terms were missing."

    band = score.band.lower()
    drivers = [t for t in score.top_drivers(3) if t.contribution < 0]
    if drivers:
        driver_text = ", ".join(t.label.split(" (")[0].lower() for t in drivers)
        tail = f"driven mainly by weak {driver_text}"
    else:
        strong = max(score.terms, key=lambda t: t.contribution)
        tail = f"supported by healthy {strong.label.split(' (')[0].lower()}"

    reliability = "" if score.is_reliable else " (note: working-capital data was unavailable)"
    return (
        f"{score.company} carries a {band} financial profile "
        f"(Altman Z'' {score.z_score:.1f}, risk {score.risk_score:.0f}/100), {tail}.{reliability}"
    )

"""Demo roster: current Indian non-financial companies, FY2026 reported financials.

Every company here has **complete data across all four market signals** as well as
financials -- news, leadership, hiring and employee sentiment. That is deliberate: these
are current, actively-covered listed companies, so the signals genuinely exist rather
than having to be reconstructed for a historical date.

All figures are from Screener.in for the year ended March 2026. Non-financial companies
only: a bank's balance sheet is out-of-distribution for an industrial ratio model.

The roster is chosen so the five companies land in *different* places and for *different
reasons* -- three distinct shapes of distress, one levered-but-improving watch case, and
two healthy names with contrasting trajectories.
"""



# ------------------------------------------------------------------ distress
SPICEJET_2026 = ScreenerFinancials(
    "SpiceJet", 2026, sales=5326, expenses=5764, operating_profit=-439, other_income=1442,
    interest=297, depreciation=645, profit_before_tax=62, net_profit=62,
    equity_capital=1413, reserves=-3356, borrowings=4219, other_liabilities=4318,
    total_assets=6594, fixed_assets=2111,
    debtor_days=8, working_capital_days=-297, ticker="SPICEJET",
)
OLA_2026 = ScreenerFinancials(
    "Ola Electric", 2026, sales=2253, expenses=3225, operating_profit=-972, other_income=207,
    interest=380, depreciation=684, profit_before_tax=-1829, net_profit=-1833,
    equity_capital=4411, reserves=-1060, borrowings=2763, other_liabilities=1674,
    total_assets=7788, fixed_assets=3352,
    debtor_days=5, working_capital_days=-185, ticker="OLAELEC",
)
VODAFONE_IDEA_2026 = ScreenerFinancials(
    "Vodafone Idea", 2026, sales=44873, expenses=25870, operating_profit=19003,
    other_income=59148, interest=21495, depreciation=22108,
    profit_before_tax=34548, net_profit=34552,
    equity_capital=108343, reserves=-144101, borrowings=192528, other_liabilities=34868,
    total_assets=191638, fixed_assets=156906,
    debtor_days=16, working_capital_days=-192, ticker="IDEA",
)

# ---------------------------------------------------------------------- watch
VEDANTA_2026 = ScreenerFinancials(
    "Vedanta", 2026, sales=78437, expenses=55254, operating_profit=23183, other_income=14186,
    interest=2817, depreciation=4810, profit_before_tax=29742, net_profit=25096,
    equity_capital=391, reserves=49261, borrowings=32947, other_liabilities=149712,
    total_assets=232311, fixed_assets=30548,
    debtor_days=6, inventory_days=66, working_capital_days=-121, ticker="VEDL",
)

# -------------------------------------------------------------------- healthy
PAYTM_2026 = ScreenerFinancials(
    "Paytm", 2026, sales=8437, expenses=7937, operating_profit=500, other_income=668,
    interest=18, depreciation=568, profit_before_tax=582, net_profit=552,
    equity_capital=64, reserves=15962, borrowings=194, other_liabilities=7695,
    total_assets=23915, fixed_assets=910,
    debtor_days=51, working_capital_days=-55, ticker="PAYTM",
)
TCS_2026 = ScreenerFinancials(
    "TCS", 2026, sales=267021, expenses=194623, operating_profit=72398, other_income=-124,
    interest=1227, depreciation=5560, profit_before_tax=65487, net_profit=49454,
    equity_capital=362, reserves=106878, borrowings=11283, other_liabilities=62644,
    total_assets=181167, fixed_assets=31343,
    debtor_days=93, working_capital_days=38, ticker="TCS",
)

#: Sector labels, defined here so the app UI and the narrative pre-warm use identical
#: strings -- the narrative cache key hashes the prompt, which includes the sector, so a
#: mismatch here silently breaks the offline cache.
SECTORS: dict[str, str] = {
    "SpiceJet": "Airline", "Ola Electric": "Electric Vehicles",
    "Vodafone Idea": "Telecom", "Vedanta": "Metals & Mining",
    "Paytm": "Fintech", "TCS": "IT Services",
}

#: (record, prior-year record or None, expected band family) for the validation gate.
ROSTER: list[tuple[ScreenerFinancials, ScreenerFinancials | None, str]] = [
    (SPICEJET_2026, None, "distress"),
    (OLA_2026, None, "distress"),
    (VODAFONE_IDEA_2026, None, "distress"),
    (VEDANTA_2026, None, "watch"),
    (PAYTM_2026, None, "healthy"),
    (TCS_2026, None, "healthy"),
]

_ACCEPT = {
    "healthy": {"Healthy"},
    "watch": {"Watch", "Elevated Risk"},
    "distress": {"Elevated Risk", "Critical"},
}


def validate_roster() -> "list[dict]":
    """Score every roster company and check it against its expected band."""

    out = []
    for fin, prior, expect in ROSTER:
        s = score_company(fin, prior=prior)
        out.append({
            "company": s.company, "year": s.year, "expected": expect,
            "z_score": round(s.z_score, 2), "risk": round(s.risk_score, 1),
            "band": s.band, "pass": s.band in _ACCEPT[expect],
        })
    return out

"""Combined Risk Score: fusion of financial and digital signals.

The product's primary output. It fuses the Altman-anchored Financial Score with the
Digital Pulse composite into one 0-100 risk score, and -- crucially -- shows the two legs
separately, because the relationship *between* them is the story.

Two design decisions, both learned the hard way earlier in the build:

1. **Renormalize over available components; never score absent digital as 0.** The current
   roster carries digital signals on every company, but the fusion must stay correct for
   any company that lacks them: a fixed 60/40 with digital=0 would drag every
   financial-only company toward healthy (0.6*fin + 0.4*0) and under-flag a distressed
   one -- the same structural-missingness bias that broke the serving path. So when
   digital is absent, combined = financial, and the UI says so.

2. **No manufactured divergence.** The flagship story is "the gap between
   the legs" (financial healthy, digital weak -> trouble not yet in the numbers). But all
   three dual-signal companies *agree* (TCS both healthy, Jet both critical, Future Retail
   both watch) -- there is no cross-sectional divergence in the real data, and fabricating
   one is exactly the mistake we corrected for Future Retail. So the narrative describes
   what is actually there: agreement (mutual confirmation) or, if it ever arises,
   divergence -- computed from the data, never assumed. The temporal thesis (digital
   leading) lives in the Jet timeline, not here.
"""


from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.signals import DigitalPulse

DEFAULT_FINANCIAL_WEIGHT = 0.60
DEFAULT_DIGITAL_WEIGHT = 0.40

# Band order, worst last, for judging whether the two legs genuinely disagree. A gap
# within the same band (Jet: financial 100, digital 77 -- both Critical) is corroboration,
# not divergence; only a band-level disagreement is worth flagging.
_BAND_RANK = {"Healthy": 0, "Watch": 1, "Elevated Risk": 2, "Critical": 3}


@dataclass
class CombinedRisk:
    company: str
    as_of: Optional[date]
    financial_score: float
    digital_score: Optional[float]      # None when digital signals are unavailable
    combined_score: float
    financial_weight: float             # actual weight used (renormalized)
    digital_weight: float
    narrative: str

    @property
    def band(self) -> str:
        return band_for(self.combined_score)

    @property
    def has_digital(self) -> bool:
        return self.digital_score is not None

    @property
    def gap(self) -> Optional[float]:
        """digital - financial. Positive => digital is the weaker (riskier) leg."""
        if self.digital_score is None:
            return None
        return round(self.digital_score - self.financial_score, 1)


def _narrative(company: str, fin: float, dig: Optional[float], combined: float) -> str:
    band = band_for(combined).lower()

    if dig is None:
        return (
            f"{company} scores {combined:.0f}/100 ({band}) on financial signals alone. "
            "Digital signals are unavailable for this company, so the combined score "
            "reflects the financial model only."
        )

    fin_rank = _BAND_RANK[band_for(fin)]
    dig_rank = _BAND_RANK[band_for(dig)]

    if fin_rank == dig_rank:
        # The real case for every dual-signal demo company: the legs agree on the band.
        return (
            f"{company} scores {combined:.0f}/100 ({band}). Financial ({fin:.0f}) and "
            f"digital ({dig:.0f}) signals agree, which raises confidence in the assessment."
        )
    if dig_rank > fin_rank:
        # Digital reads a worse band than financials -- the pattern worth surfacing.
        # Computed from the data, never assumed; only fires if it genuinely occurs.
        return (
            f"{company} scores {combined:.0f}/100 ({band}), but digital signals "
            f"({band_for(dig).lower()}, {dig:.0f}) are weaker than the financials "
            f"({band_for(fin).lower()}, {fin:.0f}) suggest. This kind of divergence can "
            "precede financial deterioration -- a signal to look closer."
        )
    return (
        f"{company} scores {combined:.0f}/100 ({band}); the risk shows more in the "
        f"financials ({band_for(fin).lower()}, {fin:.0f}) than in market signals "
        f"({band_for(dig).lower()}, {dig:.0f}) so far."
    )


def fuse(
    financial: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    financial_weight: float = DEFAULT_FINANCIAL_WEIGHT,
    digital_weight: float = DEFAULT_DIGITAL_WEIGHT,
) -> CombinedRisk:
    """Combine the two legs into the primary risk score.

    When `digital` is None the combined score is the financial score and the weights
    collapse to (1.0, 0.0) -- no phantom digital=0 term.
    """
    fin_score = financial.risk_score

    if digital is None:
        combined = fin_score
        w_fin, w_dig, dig_score = 1.0, 0.0, None
    else:
        dig_score = digital.composite_score
        total = financial_weight + digital_weight
        w_fin, w_dig = financial_weight / total, digital_weight / total
        combined = w_fin * fin_score + w_dig * dig_score

    combined = round(combined, 1)
    return CombinedRisk(
        company=financial.company,
        as_of=digital.as_of if digital is not None else None,
        financial_score=round(fin_score, 1),
        digital_score=None if dig_score is None else round(dig_score, 1),
        combined_score=combined,
        financial_weight=round(w_fin, 2),
        digital_weight=round(w_dig, 2),
        narrative=_narrative(financial.company, fin_score, dig_score, combined),
    )

"""Macroeconomic Stress Testing.

Maps macro shocks to a company's Altman-anchored risk score through explicit
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
wrong (an airline, a commodity producer and an IT firm have wildly different
pricing power and operating leverage). Defaults are conservative.

Everything is a linear recompute, so the panel updates instantly (<2s trivially).
"""


from dataclasses import dataclass, replace

from src.features import altman_z


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

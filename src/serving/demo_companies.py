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

from __future__ import annotations

from .screener import ScreenerFinancials

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
    from .financial_score import score_company

    out = []
    for fin, prior, expect in ROSTER:
        s = score_company(fin, prior=prior)
        out.append({
            "company": s.company, "year": s.year, "expected": expect,
            "z_score": round(s.z_score, 2), "risk": round(s.risk_score, 1),
            "band": s.band, "pass": s.band in _ACCEPT[expect],
        })
    return out

"""Live financials fetch: pull a listed company's latest numbers from Screener.in and
build a ScreenerFinancials the engine can score. Powers the "score any NSE company" demo.

This does a single on-demand fetch when a user asks for a company - not bulk scraping.
A production version would use a licensed data feed; this is for the live demo.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

from foresight import ScreenerFinancials

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


def _section_table(html: str, section_id: str):
    """The first <table> inside the <section id="..."> block, as a DataFrame."""
    m = re.search(rf'id=["\']{section_id}["\']', html)
    if not m:
        return None
    chunk = html[m.start():m.start() + 60000]
    tbl = re.search(r"<table.*?</table>", chunk, re.S)
    if not tbl:
        return None
    df = pd.read_html(io.StringIO(tbl.group(0)), thousands=",")[0]
    df = df.rename(columns={df.columns[0]: "item"})
    df["item"] = df["item"].astype(str).str.replace(r"[^A-Za-z%& ]", "", regex=True).str.strip()
    return df


def _market_cap(html: str) -> float:
    """Market cap in Rs Cr from Screener's top-ratios block."""
    m = re.search(r'Market Cap.*?class=["\']number["\']>\s*([\d,]+)', html, re.S)
    return float(m.group(1).replace(",", "")) if m else float("nan")


def _latest_year(df) -> str | None:
    yrs = [c for c in df.columns if re.match(r"Mar \d{4}", str(c))]
    return yrs[-1] if yrs else None


def _val(df, label: str, col) -> float:
    row = df[df["item"].str.fullmatch(label, case=False, na=False)]
    if row.empty:
        row = df[df["item"].str.startswith(label, na=False)]
    if row.empty or col is None:
        return float("nan")
    raw = str(row.iloc[0][col]).replace(",", "").replace("%", "").strip()
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def fetch_financials(ticker: str) -> ScreenerFinancials:
    """Fetch the latest fiscal-year financials for an NSE/BSE ticker from Screener."""
    tk = ticker.strip().upper()
    for path in (f"{tk}/consolidated/", f"{tk}/"):
        r = requests.get(f"https://www.screener.in/company/{path}", headers=_UA, timeout=30)
        if r.status_code == 200 and "Balance Sheet" in r.text:
            html = r.text
            break
    else:
        raise ValueError(f"Could not load Screener page for '{ticker}'.")

    pl = _section_table(html, "profit-loss")
    bs = _section_table(html, "balance-sheet")
    ra = _section_table(html, "ratios")
    if pl is None or bs is None:
        raise ValueError(f"Could not parse financial tables for '{ticker}'.")

    yp, yb, yr = _latest_year(pl), _latest_year(bs), _latest_year(ra) if ra is not None else None
    sales = _val(pl, "Sales", yp)
    if sales != sales:                       # banks/finance label it Revenue
        sales = _val(pl, "Revenue", yp)
    fin = ScreenerFinancials(
        company=tk, year=int(str(yb).split()[-1]),
        sales=sales,
        expenses=_val(pl, "Expenses", yp),
        operating_profit=_val(pl, "Operating Profit", yp),
        other_income=_val(pl, "Other Income", yp),
        interest=_val(pl, "Interest", yp),
        depreciation=_val(pl, "Depreciation", yp),
        profit_before_tax=_val(pl, "Profit before tax", yp),
        net_profit=_val(pl, "Net Profit", yp),
        equity_capital=_val(bs, "Equity Capital", yb),
        reserves=_val(bs, "Reserves", yb),
        borrowings=_val(bs, "Borrowings", yb),
        other_liabilities=_val(bs, "Other Liabilities", yb),
        total_assets=_val(bs, "Total Assets", yb),
        fixed_assets=_val(bs, "Fixed Assets", yb),
        working_capital_days=_val(ra, "Working Capital Days", yr) if ra is not None else None,
    )
    return fin, _market_cap(html)

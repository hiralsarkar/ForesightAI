"""Historical Altman-Z backtest: for real distressed companies, pull every year of
financials from Screener and compute the Altman Z each year, to see when the engine first
flagged distress relative to the actual event.

This is an honest test: the engine *is* Altman, so this does not claim to beat Altman - it
shows that the engine, run automatically on history, flagged real bankruptcies early.

Run from the repo root:  .venv/Scripts/python.exe data/indian/backtest_trajectory.py
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import requests
import foresight as f
import screener_live as S

# company -> what actually happened, for context
CASES = {
    "RCOM": "Reliance Communications - admitted to insolvency in 2019",
    "JPASSOCIAT": "Jaiprakash Associates - years of distress, NCLT in 2024",
    "SUZLON": "Suzlon Energy - near-default 2018-19, then a turnaround",
}


def multiyear(ticker: str) -> list[f.ScreenerFinancials]:
    tk = ticker.upper()
    for path in (f"{tk}/consolidated/", f"{tk}/"):
        r = requests.get(f"https://www.screener.in/company/{path}", headers=S._UA, timeout=30)
        if r.status_code == 200 and "Balance Sheet" in r.text:
            html = r.text
            break
    else:
        raise ValueError(f"could not load Screener page for {ticker}")
    pl = S._section_table(html, "profit-loss")
    bs = S._section_table(html, "balance-sheet")
    ra = S._section_table(html, "ratios")
    years = [c for c in pl.columns if re.match(r"Mar \d{4}", str(c))]
    fins = []
    for y in years:
        gp = lambda lab: S._val(pl, lab, y)
        gb = lambda lab: S._val(bs, lab, y) if (bs is not None and y in bs.columns) else float("nan")
        gr = lambda lab: S._val(ra, lab, y) if (ra is not None and y in ra.columns) else None
        fins.append(f.ScreenerFinancials(
            company=tk, year=int(y.split()[-1]),
            sales=gp("Sales"), expenses=gp("Expenses"), operating_profit=gp("Operating Profit"),
            other_income=gp("Other Income"), interest=gp("Interest"), depreciation=gp("Depreciation"),
            profit_before_tax=gp("Profit before tax"), net_profit=gp("Net Profit"),
            equity_capital=gb("Equity Capital"), reserves=gb("Reserves"), borrowings=gb("Borrowings"),
            other_liabilities=gb("Other Liabilities"), total_assets=gb("Total Assets"),
            fixed_assets=gb("Fixed Assets"), working_capital_days=gr("Working Capital Days")))
    return fins


def main():
    for tk, story in CASES.items():
        print(f"\n{tk}  ({story})")
        for fin in multiyear(tk)[-9:]:
            s = f.score_company(fin)               # book-value Altman Z (no market cap for history)
            flag = "  <-- distress" if s.zone == "Distress" else ""
            print(f"   FY{fin.year}   Altman Z {s.z_score:7.2f}   {s.zone}{flag}")


if __name__ == "__main__":
    main()

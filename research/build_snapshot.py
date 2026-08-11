"""Bake a financials snapshot for the whole scoreable NSE universe.

Run offline / at build time -- NOT in the hosted app. Fetches each ticker's latest-year
financials + market cap live from Screener once, and writes them to
`data/financials_snapshot.json`. At runtime `screener_live.fetch_financials` reads that
file, so the deployed app scores any listed name without an outbound call to screener.in
(the source of the '[Errno 111] Connection refused' failures on the sandboxed host).

Usage:
    .venv/Scripts/python.exe research/build_snapshot.py

Idempotent: re-run whenever the NSE universe or the source financials change.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import screener_live
from nse_universe import NSE_TOP

_OUT = _ROOT / "data" / "financials_snapshot.json"


def main() -> int:
    companies: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []

    for i, (name, ticker) in enumerate(sorted(NSE_TOP.items()), 1):
        tk = ticker.strip().upper()
        try:
            fin, mcap = screener_live.fetch_live(tk)
            companies[tk] = {"name": name, "fin": dataclasses.asdict(fin), "market_cap": mcap}
            print(f"[{i:>2}/{len(NSE_TOP)}] {name:<26} {tk:<12} ok  mcap={mcap:,.0f}")
        except Exception as exc:  # keep going; report the misses at the end
            failures.append((f"{name} ({tk})", str(exc)))
            print(f"[{i:>2}/{len(NSE_TOP)}] {name:<26} {tk:<12} FAIL {exc}")
        time.sleep(1.0)  # be polite to Screener

    payload = {
        "generated": date.today().isoformat(),
        "source": "screener.in (consolidated where available)",
        "count": len(companies),
        "companies": companies,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(companies)}/{len(NSE_TOP)} companies to {_OUT.relative_to(_ROOT)}")
    if failures:
        print(f"{len(failures)} failed:")
        for who, why in failures:
            print(f"  - {who}: {why}")
    return 0 if companies else 1


if __name__ == "__main__":
    raise SystemExit(main())

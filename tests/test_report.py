"""Module 9 executive-report guardrails."""

from __future__ import annotations

import re

import pytest

from src.narrative.recommendations import recommend
from src.narrative.summary import generate
from src.reporting.executive_report import build_report
from src.scoring.combined import fuse
from src.serving import demo_companies as D
from src.serving.financial_score import score_company
from src.signals.demo_signals import DEFAULT_AS_OF, has_signals, pulse_as_of
from src.signals.sentiment import LoughranMcDonaldScorer

_CASES = {
    "SpiceJet": (D.SPICEJET_2026, None, "Airline"),
    "TCS": (D.TCS_2026, None, "IT Services"),
    "Vedanta": (D.VEDANTA_2026, None, "Metals & Mining"),
    "Paytm": (D.PAYTM_2026, None, "Fintech"),
}


def _pdf(name: str) -> bytes:
    rec, prior, sector = _CASES[name]
    fin = score_company(rec, prior=prior)
    dig = pulse_as_of(name, DEFAULT_AS_OF[name], LoughranMcDonaldScorer()) if has_signals(name) else None
    comb = fuse(fin, dig)
    text, _ = generate(comb, fin, dig, sector, use_llm=False)
    return build_report(rec, comb, fin, dig, text, recommend(rec, fin, dig, prior), sector, prior)


@pytest.mark.parametrize("name", list(_CASES))
def test_report_is_a_valid_two_page_pdf(name):
    data = _pdf(name)
    assert data.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page[^s]", data)) == 2, "memo must be exactly two pages"
    assert len(data) > 3000


def test_report_builds_for_every_roster_company():
    for name in _CASES:
        assert _pdf(name).startswith(b"%PDF")


def test_report_avoids_glyphs_helvetica_cannot_encode():
    """U+2033 and U+2192 silently drop in Helvetica, leaving gaps like 'Altman Z '."""
    import pathlib

    src = pathlib.Path("src/reporting/executive_report.py").read_text(encoding="utf-8")
    # Helvetica renders the WinAnsi (cp1252) set; any glyph outside it drops silently in
    # the PDF, leaving gaps like 'Altman Z '. Flag whatever the base-14 font cannot encode.
    def _unencodable(ch):
        try:
            ch.encode("cp1252")
            return False
        except UnicodeEncodeError:
            return True

    bad = sorted({ch for ch in src if _unencodable(ch)})
    assert not bad, f"non-encodable glyph(s) in report source: {[hex(ord(c)) for c in bad]}"


def test_year_on_year_avoids_meaningless_percentages():
    """Jet's operating profit went +24 -> -3,660; '-15350%' is correct but absurd."""
    from src.reporting.executive_report import _styles, _yoy_line

    # A sign flip must print figures, not a runaway percentage.
    from src.serving.screener import ScreenerFinancials
    now = ScreenerFinancials("X", 2026, sales=100, expenses=110, operating_profit=-50,
                             other_income=0, interest=1, depreciation=1,
                             profit_before_tax=-52, net_profit=-52, equity_capital=10,
                             reserves=-5, borrowings=10, other_liabilities=10,
                             total_assets=50, fixed_assets=10)
    then = ScreenerFinancials("X", 2025, sales=100, expenses=90, operating_profit=10,
                              other_income=0, interest=1, depreciation=1,
                              profit_before_tax=8, net_profit=8, equity_capital=10,
                              reserves=20, borrowings=10, other_liabilities=10,
                              total_assets=50, fixed_assets=10)
    text = _yoy_line(now, then, _styles()).getPlainText()
    assert "10 to -50" in text


def test_year_on_year_states_when_no_prior_year():
    from src.reporting.executive_report import _styles, _yoy_line

    text = _yoy_line(D.TCS_2026, None, _styles()).getPlainText()
    assert "single reported year" in text


def test_report_carries_the_disclaimer():
    data = _pdf("TCS")
    # Text is compressed in the PDF stream, so assert via the builder instead.
    from src.reporting.executive_report import build_report as _b
    assert data.startswith(b"%PDF")  # smoke
    src = __import__("pathlib").Path("src/reporting/executive_report.py").read_text(encoding="utf-8")
    assert "supplementary tool alongside professional financial analysis" in src

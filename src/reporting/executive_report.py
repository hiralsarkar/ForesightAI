"""Module 9 -- Executive Report: a two-page credit-intelligence memo (ReportLab).

Page 1  identity, combined risk score, key financial metrics, market signals, narrative.
Page 2  Altman decomposition, recommendations by audience, disclaimer.

Two deliberate departures from the blueprint's page-2 spec, both evidence-driven:

* It asks for a SHAP waterfall. Our *serving* explanation is the exact Altman Z'' 4-term
  decomposition (the GBM does not drive the displayed score), so that is what we print.
* It asks for debt and profitability *trajectory* charts. We curated a single fiscal year
  for seven of eight companies, so multi-year trajectories do not exist for them and are
  not invented. Where a prior year exists the year-on-year move is printed as a line of
  text instead.

The closing disclaimer is not weakness -- real enterprise risk reporting carries one.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

from ..narrative.recommendations import Audience, Recommendation
from ..scoring.combined import CombinedRisk
from ..serving.financial_score import FinancialScore
from ..serving.screener import ScreenerFinancials, compute_features
from ..signals.composite import DigitalPulse

# Palette mirrors the dashboard so the report is recognisably the same product.
NAVY = colors.HexColor("#0A1628")
PANEL = colors.HexColor("#0F1E33")
AMBER = colors.HexColor("#F59E0B")
DIM = colors.HexColor("#94A3B8")
BORDER = colors.HexColor("#1E3350")
GOOD = colors.HexColor("#22C55E")
WATCH = colors.HexColor("#F59E0B")
ELEVATED = colors.HexColor("#F97316")
BAD = colors.HexColor("#EF4444")

_BAND_COLOR = {"Healthy": GOOD, "Watch": WATCH, "Elevated Risk": ELEVATED, "Critical": BAD}

_PAGE_W, _PAGE_H = A4
_MARGIN = 16 * mm
_BANNER_H = 22 * mm


def _band_color(band: str):
    return _BAND_COLOR.get(band, DIM)


# ------------------------------------------------------------------------ styles
def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9, leading=13,
                          textColor=colors.HexColor("#1F2937"), alignment=TA_LEFT)
    return {
        "body": base,
        "section": ParagraphStyle("section", parent=base, fontName="Helvetica-Bold",
                                  fontSize=8, textColor=colors.HexColor("#64748B"),
                                  spaceAfter=4, leading=11),
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=17,
                             textColor=NAVY, leading=20),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9, textColor=colors.HexColor("#64748B")),
        "narr": ParagraphStyle("narr", parent=base, fontSize=9.5, leading=14),
        "rec_t": ParagraphStyle("rec_t", parent=base, fontName="Helvetica-Bold", fontSize=8.5,
                                leading=11),
        "rec_b": ParagraphStyle("rec_b", parent=base, fontSize=8, leading=11,
                                textColor=colors.HexColor("#475569")),
        "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=DIM),
        "disc": ParagraphStyle("disc", parent=base, fontSize=7.5, leading=10,
                               textColor=colors.HexColor("#64748B")),
    }


# ------------------------------------------------------------- page furniture
def _draw_page(canvas, doc, company: str, generated: str) -> None:
    canvas.saveState()
    # Navy banner with the brand.
    canvas.setFillColor(NAVY)
    canvas.rect(0, _PAGE_H - _BANNER_H, _PAGE_W, _BANNER_H, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(_MARGIN, _PAGE_H - 14 * mm, "Foresight")
    canvas.setFillColor(AMBER)
    canvas.drawString(_MARGIN + 24 * mm, _PAGE_H - 14 * mm, "AI")
    canvas.setFillColor(DIM)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(_PAGE_W - _MARGIN, _PAGE_H - 14 * mm,
                           "CREDIT INTELLIGENCE MEMO")
    # Footer.
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(_MARGIN, 10 * mm, f"{company}  |  Generated {generated}")
    canvas.drawRightString(_PAGE_W - _MARGIN, 10 * mm, f"Page {canvas.getPageNumber()} of 2")
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.line(_MARGIN, 13 * mm, _PAGE_W - _MARGIN, 13 * mm)
    canvas.restoreState()


# ------------------------------------------------------------------- components
def _score_block(combined: CombinedRisk, fin: FinancialScore, st) -> Table:
    """Large coloured score box plus the two component legs."""
    c = _band_color(combined.band)
    score_cell = Table(
        [[Paragraph(f'<font size="30" color="white"><b>{combined.combined_score:.0f}</b></font>'
                    f'<font size="11" color="white">/100</font>', st["body"])],
         [Paragraph(f'<font size="10" color="white"><b>{combined.band.upper()}</b></font>', st["body"])]],
        colWidths=[46 * mm], rowHeights=[16 * mm, 7 * mm])
    score_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    dig = f"{combined.digital_score:.0f}/100" if combined.has_digital else "Not available"
    detail = Table([
        ["Financial Health", f"{combined.financial_score:.0f}/100",
         f"weight {combined.financial_weight:.0%}"],
        ["Market Signals", dig, f"weight {combined.digital_weight:.0%}"],
        ["Altman Z''", f"{fin.z_score:.2f}", f"{fin.zone} zone"],
    ], colWidths=[34 * mm, 26 * mm, 32 * mm])
    detail.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("FONT", (1, 0), (1, -1), "Helvetica-Bold", 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#94A3B8")),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    wrap = Table([[score_cell, detail]], colWidths=[50 * mm, 96 * mm])
    wrap.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (0, 0), 0)]))
    return wrap


def _metrics_table(rec: ScreenerFinancials, prior, st) -> Table:
    f = compute_features(rec, prior=prior)

    def fmt(key, pct=False, suffix=""):
        v = f.get(key, float("nan"))
        if v != v:
            return "n/a", DIM
        txt = f"{v*100:.0f}%" if pct else f"{v:.2f}{suffix}"
        return txt, None

    rows = [
        ("Interest Coverage", *fmt("Attr27", suffix="x"), lambda v: v >= 2, "Attr27"),
        ("Debt to Assets", *fmt("Attr2", pct=True), lambda v: v < 0.7, "Attr2"),
        ("Return on Assets", *fmt("Attr1", pct=True), lambda v: v > 0, "Attr1"),
        ("Equity Ratio", *fmt("Attr10", pct=True), lambda v: v > 0.2, "Attr10"),
    ]
    data, styles = [[], []], []
    for i, (label, txt, _dim, ok, key) in enumerate(rows):
        v = f.get(key, float("nan"))
        col = DIM if v != v else (GOOD if ok(v) else BAD)
        data[0].append(Paragraph(f'<font size="7" color="#64748B">{label.upper()}</font>', st["body"]))
        data[1].append(Paragraph(f'<font size="13" color="{col.hexval()}"><b>{txt}</b></font>', st["body"]))
    t = Table(data, colWidths=[36.5 * mm] * 4)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _signals_block(digital: Optional[DigitalPulse], st) -> list:
    if digital is None:
        return [Paragraph("Market-intelligence signals are not available for this company; "
                          "the assessment rests on reported financials alone.", st["body"])]
    out = []
    for r in digital.readings:
        col = _band_color(r.band)
        out.append(Paragraph(
            f'<font color="{col.hexval()}">•</font> <b>{r.kind.value}</b> '
            f'<font color="#64748B">({r.label})</font> - {r.datum}', st["body"]))
        out.append(Spacer(1, 2.5 * mm))
    return out


def _decomposition(fin: FinancialScore, st) -> Table:
    """Exact Altman Z'' 4-term decomposition -- the terms sum to the score."""
    maxc = max((abs(t.contribution) for t in fin.terms), default=1.0) or 1.0
    data = []
    for t in fin.terms:
        col = GOOD if t.contribution > 0 else BAD
        width = max(1.0, abs(t.contribution) / maxc * 42)
        bar = Table([[""]], colWidths=[width * mm], rowHeights=[3.4 * mm])
        bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        data.append([
            Paragraph(f'<font size="8">{t.label}</font>', st["body"]), bar,
            Paragraph(f'<font size="8" color="{col.hexval()}"><b>{t.contribution:+.2f}</b></font>'
                      f'<font size="7" color="#94A3B8">  ({t.value:+.2f} &times; {t.coefficient})</font>',
                      st["body"]),
        ])
    data.append([Paragraph('<font size="8"><b>Altman Z'' (sum)</b></font>', st["body"]), "",
                 Paragraph(f'<font size="8"><b>{fin.z_score:+.2f}</b></font>', st["body"])])
    t = Table(data, colWidths=[62 * mm, 44 * mm, 40 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    return t


def _recs_block(recs: dict[Audience, list[Recommendation]], st) -> list:
    out = []
    for aud in (Audience.LENDER, Audience.MANAGEMENT):
        items = recs.get(aud, [])
        out.append(Paragraph(f'<font color="#B45309"><b>{aud.value.upper()}</b></font>', st["section"]))
        out.append(Spacer(1, 1.5 * mm))
        if not items:
            out.append(Paragraph("No specific actions indicated.", st["rec_b"]))
        for it in items[:4]:  # keep the memo to two pages
            out.append(Paragraph(f"{it.title}", st["rec_t"]))
            out.append(Paragraph(it.action, st["rec_b"]))
            out.append(Spacer(1, 2 * mm))
        out.append(Spacer(1, 3 * mm))
    return out


def _yoy_line(rec: ScreenerFinancials, prior: Optional[ScreenerFinancials], st):
    """Year-on-year move where a prior year exists; otherwise say so plainly."""
    if prior is None:
        return Paragraph("Only a single reported year was available for this company, so a "
                         "multi-year trajectory is not shown.", st["small"])

    def move(now, then):
        """Percentage change, but fall back to absolute values when a percentage would
        mislead. Jet's operating profit went +24 -> -3,660: '-15350%' is arithmetically
        correct and completely meaningless to a reader, so print the figures instead."""
        if not then:
            return f"{then:,.0f} to {now:,.0f}"
        if (now < 0) != (then < 0):          # sign flip
            return f"{then:,.0f} to {now:,.0f}"
        change = (now - then) / abs(then) * 100
        if abs(change) > 300:                 # tiny base -> runaway percentage
            return f"{then:,.0f} to {now:,.0f}"
        return f"{change:+.0f}%"

    return Paragraph(
        f"Year on year ({prior.year} to {rec.year}): sales {move(rec.sales, prior.sales)}, "
        f"operating profit {move(rec.operating_profit, prior.operating_profit)}, "
        f"borrowings {move(rec.borrowings, prior.borrowings)}, "
        f"reserves {move(rec.reserves, prior.reserves)}. Figures in Rs crore.", st["small"])


# ------------------------------------------------------------------ public API
def build_report(
    rec: ScreenerFinancials,
    combined: CombinedRisk,
    fin: FinancialScore,
    digital: Optional[DigitalPulse],
    narrative: str,
    recommendations: dict[Audience, list[Recommendation]],
    sector: str = "",
    prior: Optional[ScreenerFinancials] = None,
) -> bytes:
    """Render the two-page memo and return the PDF bytes."""
    st = _styles()
    generated = datetime.now().strftime("%d %b %Y %H:%M")
    buf = BytesIO()

    doc = BaseDocTemplate(
        buf, pagesize=A4, leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_BANNER_H + 8 * mm, bottomMargin=18 * mm,
        title=f"Foresight AI - {rec.company}", author="Foresight AI Analytics Engine",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(
        id="main", frames=[frame],
        onPage=lambda c, d: _draw_page(c, d, rec.company, generated))])

    story: list = []
    # --- page 1 ---------------------------------------------------------
    story.append(Paragraph(rec.company, st["h1"]))
    meta = f"{sector} &middot; " if sector else ""
    story.append(Paragraph(f"{meta}FY{rec.year} reported financials &middot; "
                           f"Analyst: Foresight AI Analytics Engine", st["sub"]))
    story.append(Spacer(1, 5 * mm))
    story.append(_score_block(combined, fin, st))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("KEY FINANCIAL METRICS", st["section"]))
    story.append(_metrics_table(rec, prior, st))
    story.append(Spacer(1, 2.5 * mm))
    story.append(_yoy_line(rec, prior, st))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("MARKET INTELLIGENCE SIGNALS", st["section"]))
    story.extend(_signals_block(digital, st))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("ANALYST SUMMARY", st["section"]))
    story.append(Paragraph(narrative, st["narr"]))

    # --- page 2 ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("WHY THIS SCORE - ALTMAN Z'' DECOMPOSITION", st["section"]))
    story.append(Spacer(1, 1.5 * mm))
    story.append(_decomposition(fin, st))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("The four terms are exact and sum to the Z'' score; green "
                           "reduces risk, red increases it.", st["small"]))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("RECOMMENDED ACTIONS", st["section"]))
    story.append(Spacer(1, 1.5 * mm))
    story.extend(_recs_block(recommendations, st))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "This report is generated by an AI analytics system and should be used as a "
        "supplementary tool alongside professional financial analysis. Financial scores "
        "are derived from reported financial statements using the Altman Z'' model. "
        "Market-intelligence signals may include illustrative data and are labelled where "
        "so; they are not a substitute for verified disclosure.", st["disc"]))

    doc.build(story)
    return buf.getvalue()

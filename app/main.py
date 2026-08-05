"""Foresight AI -- Corporate Financial Health Intelligence dashboard.

Three views (tabs):
  * Company Analysis -- the demo spine: combined gauge, financial cards,
    digital pulse, Altman waterfall, stress test.
  * Portfolio Monitor -- surveillance table across the roster. The hero
    visual: verified financials, clean red-to-green contrast.
  * Case Study -- Digital Pulse over the last year for a company under active stress,
    against its annual financial score.

Panels use the `panel()` helper (a keyed `st.container(border=True)`) rather than a bare
`<div class="fa-panel">`: a standalone opening div is sanitized by Streamlit into an empty
box, so the styled frame never wraps the widgets that follow. The container key gives
`theme.py` a stable `.st-key-fapanel-*` hook to style.

Run: streamlit run app/main.py --server.fileWatcherType none
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                 # app/ - scoring_service, theme
sys.path.insert(0, str(_HERE.parent / "src"))  # src/ - foresight

import plotly.graph_objects as go
import streamlit as st

import scoring_service as svc
import theme
from theme import band_color, band_pill

st.set_page_config(page_title="Foresight AI", page_icon="\U0001F4C8", layout="wide")
theme.inject()


_panel_seq = 0


def panel():
    """A bordered container carrying a stable `fapanel-*` key so theme.py can style it.

    Keys must be unique within a run; the counter resets each Streamlit rerun (the script
    re-executes top to bottom), so the same panels get the same keys every time.
    """
    global _panel_seq
    _panel_seq += 1
    return st.container(border=True, key=f"fapanel-{_panel_seq}")


def section_title(text: str) -> None:
    st.markdown(f'<div class="fa-section-title">{text}</div>', unsafe_allow_html=True)


# =============================================================================== gauge
def risk_gauge(score: float, band: str) -> go.Figure:
    color = band_color(band)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 46, "color": theme.TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": theme.TEXT_DIM,
                     "tickfont": {"color": theme.TEXT_DIM, "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": theme.BG_PANEL_2, "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "#14311F"}, {"range": [25, 50], "color": "#3A2F13"},
                {"range": [50, 75], "color": "#3A2412"}, {"range": [75, 100], "color": "#3A1717"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": theme.TEXT})
    return fig


def component_bar(label: str, value, weight: float) -> str:
    if value is None:
        return (f'<div style="margin:8px 0"><span style="color:{theme.TEXT_DIM};font-size:0.8rem">'
                f'{label}</span><div style="color:{theme.TEXT_DIM};font-size:0.9rem">'
                f'Unavailable - combined reflects financials only</div></div>')
    from foresight import band_for
    c = band_color(band_for(value))
    pct = max(2, min(100, value))
    return (
        f'<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;'
        f'font-size:0.8rem"><span style="color:{theme.TEXT_DIM}">{label} &middot; weight '
        f'{weight:.0%}</span><span style="color:{c};font-weight:700">{value:.0f}</span></div>'
        f'<div style="background:{theme.BG_BASE};border-radius:4px;height:8px;margin-top:4px">'
        f'<div style="width:{pct}%;background:{c};height:8px;border-radius:4px"></div></div></div>'
    )


# =================================================================== Company Analysis
def render_company(company: str) -> None:
    c = svc.combined(company)
    fin = svc.financial(company)
    dig = svc.digital(company)

    g1, g2 = st.columns([2, 3])
    with g1:
        with panel():
            section_title("Combined Risk Score")
            st.plotly_chart(risk_gauge(c.combined_score, c.band), width='stretch',
                            config={"displayModeBar": False})
            st.markdown(f'<div style="text-align:center;margin-top:-10px">{band_pill(c.band)}</div>',
                        unsafe_allow_html=True)
    with g2:
        with panel():
            section_title("Contributing Signals")
            st.markdown(component_bar("Financial Health", c.financial_score, c.financial_weight),
                        unsafe_allow_html=True)
            st.markdown(component_bar("Digital Signals", c.digital_score, c.digital_weight),
                        unsafe_allow_html=True)
            st.markdown(f'<div class="fa-narrative" style="margin-top:14px">{c.narrative}</div>',
                        unsafe_allow_html=True)

    # financial cards
    with panel():
        section_title(f'Financial Health - Altman Z&Prime; {fin.z_score:.1f} '
                      f'&middot; {band_pill(fin.band)}')
        cards = svc.ratio_cards(company)
        for col, card in zip(st.columns(len(cards)), cards):
            color = theme.GOOD if card.good else theme.BAD
            col.markdown(f'<div class="fa-card"><div class="lbl">{card.label}</div>'
                         f'<div class="val" style="color:{color}">{card.value}</div>'
                         f'<div class="ctx">{card.context}</div></div>', unsafe_allow_html=True)

    # digital pulse
    with panel():
        section_title("Market Intelligence Signals")
        if dig is None:
            st.markdown(f'<div style="color:{theme.TEXT_DIM}">Digital signals are not available for '
                        f'{company}. The combined score reflects financial data only.</div>',
                        unsafe_allow_html=True)
        else:
            for r in dig.readings:
                c2 = band_color(r.band)
                st.markdown(f'<div class="fa-signal"><div class="fa-dot" style="background:{c2}"></div>'
                            f'<div class="name">{r.kind.value}</div><div class="datum">{r.datum}</div>'
                            f'<div class="pill" style="background:{c2}22;color:{c2}">{r.label}</div></div>',
                            unsafe_allow_html=True)

    # Altman waterfall
    with panel():
        section_title("Why This Score - Altman Z&Prime; Decomposition")
        maxc = max((abs(t.contribution) for t in fin.terms), default=1.0)
        for t in fin.terms:
            c3 = theme.GOOD if t.contribution > 0 else theme.BAD
            width = int(abs(t.contribution) / maxc * 220)
            st.markdown(f'<div class="fa-term"><div class="tl">{t.label}</div>'
                        f'<div class="bar" style="width:{width}px;background:{c3}"></div>'
                        f'<div class="tv">{t.contribution:+.2f} &nbsp; ({t.value:+.2f} &times; {t.coefficient})</div>'
                        f'</div>', unsafe_allow_html=True)
        from foresight import narrative as fin_narrative
        st.markdown(f'<div class="fa-narrative" style="margin-top:12px">{fin_narrative(fin)}</div>',
                    unsafe_allow_html=True)

    # report download
    st.download_button(
        "Generate Executive Report (PDF)", data=svc.report_pdf(company),
        file_name=f"ForesightAI_{company.replace(' ', '_')}.pdf", mime="application/pdf",
        key=f"dl_{company}")

    # AI narrative
    text, source = svc.narrative(company)
    tag = ("AI-generated" if source == "llm" else "Rule-based (LLM unavailable)")
    with panel():
        st.markdown(f'<div class="fa-section-title">Analyst Summary '
                    f'<span style="color:{theme.TEXT_DIM};text-transform:none;letter-spacing:0">'
                    f'&middot; {tag}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fa-narrative" style="font-size:0.95rem;line-height:1.6">{text}</div>',
                    unsafe_allow_html=True)

    # recommendations, split by audience
    recs = svc.advice(company)
    with panel():
        section_title("Recommended Actions")
        cols = st.columns(2)
        for col, (aud, items) in zip(cols, recs.items()):
            with col:
                st.markdown(f'<div style="color:{theme.ACCENT};font-weight:700;font-size:0.8rem;'
                            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px">'
                            f'{aud.value}</div>', unsafe_allow_html=True)
                if not items:
                    st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.85rem">'
                                'No specific actions indicated.</div>', unsafe_allow_html=True)
                for it in items:
                    accent = theme.BAD if it.priority <= 1 else theme.WATCH if it.priority <= 3 else theme.TEXT_DIM
                    st.markdown(
                        f'<div style="background:{theme.BG_PANEL_2};border-left:3px solid {accent};'
                        f'border-radius:6px;padding:10px 14px;margin-bottom:8px">'
                        f'<div style="font-weight:600;font-size:0.88rem">{it.title}</div>'
                        f'<div style="color:{theme.TEXT_DIM};font-size:0.8rem;line-height:1.45;'
                        f'margin-top:4px">{it.action}</div></div>', unsafe_allow_html=True)

    # macro + company stress test
    with panel():
        section_title("Scenario Analysis - Stress the Macro Environment")
        st.markdown(f'<div style="color:{theme.ACCENT};font-weight:700;font-size:0.72rem;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px">'
                    'Macroeconomic scenario</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        d_int = m1.slider("Interest rate change (bps, sustained 3 yr)", 0, 400, 0, 25, key=f"int_{company}")
        d_infl = m2.slider("Inflation change (pp)", 0, 6, 0, 1, key=f"infl_{company}")
        d_gdp = m3.slider("GDP growth change (pp)", -5, 3, 0, 1, key=f"gdp_{company}")
        st.markdown(f'<div style="color:{theme.ACCENT};font-weight:700;font-size:0.72rem;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin:8px 0 2px">'
                    'Company-specific shocks</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        op_shock = c1.slider("Operating profit shock (%)", -50, 20, 0, 5, key=f"op_{company}")
        lev_shock = c2.slider("Leverage shock (%)", -20, 50, 0, 5, key=f"lev_{company}")

        res = svc.stress(company, op_shock, lev_shock, interest_bps=d_int,
                         inflation_pp=d_infl, gdp_pp=d_gdp)
        dc = theme.BAD if res.delta > 0 else theme.GOOD if res.delta < 0 else theme.TEXT_DIM

        def _cov(x):
            return "n/a" if x != x else f"{x:.1f}x"
        st.markdown(
            f'<div style="display:flex;gap:30px;align-items:baseline;margin-top:10px;flex-wrap:wrap">'
            f'<div><span style="color:{theme.TEXT_DIM};font-size:0.8rem">BASE</span><br>'
            f'<span style="font-size:1.6rem;font-weight:700">{res.base_score:.0f}</span></div>'
            f'<div><span style="color:{theme.TEXT_DIM};font-size:0.8rem">STRESSED</span><br>'
            f'<span style="font-size:1.6rem;font-weight:700">{res.stressed_score:.0f}</span></div>'
            f'<div><span style="color:{theme.TEXT_DIM};font-size:0.8rem">CHANGE</span><br>'
            f'<span style="font-size:1.6rem;font-weight:700;color:{dc}">{res.delta:+.0f}</span></div>'
            f'<div><span style="color:{theme.TEXT_DIM};font-size:0.8rem">INTEREST COVERAGE</span><br>'
            f'<span style="font-size:1.15rem;font-weight:700">{_cov(res.base_coverage)} '
            f'<span style="color:{theme.TEXT_DIM}">&rarr;</span> {_cov(res.stressed_coverage)}</span></div>'
            f'<div style="color:{theme.TEXT_DIM};font-size:0.82rem;align-self:center">'
            f'Altman Z&Prime; {res.base_z} &rarr; {res.stressed_z}</div></div>', unsafe_allow_html=True)

        # Attribution: which lever drove the move.
        active = {k: v for k, v in res.contributions.items() if abs(v) > 0.05}
        if active:
            parts = " &middot; ".join(f'{k} <b style="color:{dc}">{v:+.1f}</b>'
                                      for k, v in sorted(active.items(), key=lambda x: -abs(x[1])))
            st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.8rem;margin-top:10px">'
                        f'Score impact by driver: {parts}</div>', unsafe_allow_html=True)


# ==================================================================== Portfolio
def render_portfolio() -> None:
    rows = svc.portfolio()
    with panel():
        section_title(f'Portfolio Risk Monitor - {len(rows)} companies, highest risk first')

        grid = "1.7fr 1.1fr 0.85fr 0.85fr 0.85fr 1.15fr"
        st.markdown(
            f'<div style="display:grid;grid-template-columns:{grid};'
            f'gap:8px;padding:0 12px 8px;color:{theme.TEXT_DIM};font-size:0.72rem;'
            'text-transform:uppercase;letter-spacing:0.05em">'
            '<div>Company</div><div>Sector</div><div>Combined</div><div>Financial</div>'
            '<div>Digital</div><div>Status</div></div>',
            unsafe_allow_html=True)

        for r in rows:
            c = band_color(r.band)
            st.markdown(
                f'<div class="fa-row" style="display:grid;grid-template-columns:{grid};'
                f'gap:8px;padding:12px;margin-bottom:6px;border-radius:8px;align-items:center;'
                f'background:{c}14;border-left:4px solid {c}">'
                f'<div style="font-weight:600">{r.company}</div>'
                f'<div style="color:{theme.TEXT_DIM};font-size:0.85rem">{r.sector}</div>'
                f'<div style="font-weight:800;font-size:1.15rem;color:{c}">{r.combined:.0f}</div>'
                f'<div style="color:{theme.TEXT}">{r.financial:.0f}</div>'
                f'<div style="color:{theme.TEXT}">{r.digital:.0f}</div>'
                f'<div>{band_pill(r.band)}</div></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="color:{theme.TEXT_DIM};font-size:0.8rem;margin-top:6px">Financial scores '
            'are computed from FY2026 reported financials (Altman Z&Prime;). Digital scores fuse news '
            'sentiment, leadership changes, hiring trend and employee sentiment for the same '
            'period.</div>', unsafe_allow_html=True)


# ================================================================== Case Study (timeline)
def render_case_study() -> None:
    company = "Ola Electric"
    tl = svc.case_timeline(company)
    labels = [p.label for p in tl]

    fig = go.Figure()
    # Band zones as background shapes.
    for lo, hi, col in [(0, 25, "#14311F"), (25, 50, "#3A2F13"), (50, 75, "#3A2412"), (75, 100, "#3A1717")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.5, line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=labels, y=[p.financial for p in tl], name="Financial (FY2026 filing)",
        mode="lines+markers", line=dict(color=theme.TEXT_DIM, width=2, dash="dot"),
        marker=dict(size=8)))
    fig.add_trace(go.Scatter(
        x=labels, y=[p.digital for p in tl], name="Digital Pulse (live signals)",
        mode="lines+markers", line=dict(color=theme.ACCENT, width=3),
        marker=dict(size=10)))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme.TEXT}, yaxis=dict(range=[0, 100], title="Risk score", gridcolor=theme.BORDER),
        xaxis=dict(gridcolor=theme.BORDER),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    with panel():
        section_title(f'{company} - Signals Ahead of the Filing')
        st.markdown('<div class="fa-headline">The annual accounts update once. The signals moved '
                    'every month.</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown(
            '<div class="fa-narrative">The FY2026 accounts are a single annual data point. '
            'The market signals moved through the year: the chief technology and marketing '
            'officers left in late 2025, the chief financial officer resigned on 19 January '
            '2026, the company cut 5% of its workforce on 31 January, and retail '
            'registrations stayed below 10,000 units for a third consecutive month while a '
            'rival booked 20,786. Each of those was observable well before the annual filing '
            'that confirmed them.</div>', unsafe_allow_html=True)


# ==================================================================== Review Economics
def render_economics() -> None:
    with panel():
        section_title("Review Economics - the cost-optimal policy")
        st.markdown('<div class="fa-headline">Put a price on a miss and a review; the model '
                    'tells you how deep to look.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        miss = c1.slider("Cost of a missed distress (Rs lakh)", 10, 500, 50, 10, key="ec_miss")
        rev = c2.slider("Cost of one review (Rs lakh)", 1, 20, 1, 1, key="ec_rev")

        e = svc.economics(float(miss), float(rev))
        of, oa = e["opt_foresight"], e["opt_altman"]
        pos = e["total_distress"]
        saved = oa["cost_cr"] - of["cost_cr"]
        pct_saved = saved / oa["cost_cr"] * 100 if oa["cost_cr"] else 0

        def stat(lbl, val, color=theme.TEXT):
            return (f'<div><span style="color:{theme.TEXT_DIM};font-size:0.72rem;'
                    f'text-transform:uppercase;letter-spacing:0.05em">{lbl}</span><br>'
                    f'<span style="font-size:1.5rem;font-weight:700;color:{color}">{val}</span></div>')

        st.markdown(
            f'<div style="display:flex;gap:34px;flex-wrap:wrap;margin-top:14px;align-items:flex-end">'
            + stat("Recommended review", f'{of["budget_pct"]:.0f}%', theme.ACCENT)
            + stat("Firms", f'{of["reviewed"]:,}')
            + stat("Distress caught", f'{of["catch_rate"]*100:.0f}%')
            + stat("Expected cost", f'Rs {of["cost_cr"]:.1f} cr')
            + stat("vs Altman best", f'Rs {oa["cost_cr"]:.1f} cr', theme.TEXT_DIM)
            + stat("Saving", f'Rs {saved:.1f} cr ({pct_saved:.0f}%)', theme.GOOD)
            + '</div>', unsafe_allow_html=True)

    with panel():
        section_title("Expected cost by review budget")
        fx = [r["budget_pct"] for r in e["foresight"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fx, y=[r["cost_cr"] for r in e["foresight"]],
                                 mode="lines", name="Foresight AI",
                                 line=dict(color=theme.ACCENT, width=3)))
        fig.add_trace(go.Scatter(x=fx, y=[r["cost_cr"] for r in e["altman"]],
                                 mode="lines", name="Altman Z'' screen",
                                 line=dict(color=theme.TEXT_DIM, width=2, dash="dash")))
        for opt, col in ((of, theme.ACCENT), (oa, theme.TEXT_DIM)):
            fig.add_trace(go.Scatter(x=[opt["budget_pct"]], y=[opt["cost_cr"]], mode="markers",
                                     marker=dict(color=col, size=13, line=dict(color="white", width=1)),
                                     showlegend=False))
        fig.add_annotation(x=of["budget_pct"], y=of["cost_cr"], text=f"optimal {of['budget_pct']:.0f}%",
                           font=dict(color=theme.ACCENT, size=12), showarrow=True, arrowcolor=theme.ACCENT,
                           ax=30, ay=-30)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color": theme.TEXT},
                          xaxis=dict(title="Review budget (% of portfolio)", gridcolor=theme.BORDER),
                          yaxis=dict(title="Expected cost (Rs cr)", gridcolor=theme.BORDER),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown(
            f'<div class="fa-narrative">On the validation portfolio ({e["n"]:,} firms, {pos} in '
            f'distress), ranking by risk and reviewing the cost-optimal {of["budget_pct"]:.0f}% catches '
            f'{of["caught"]}/{pos} distress for Rs {of["cost_cr"]:.1f} cr. The Altman screen\'s best '
            f'policy costs Rs {oa["cost_cr"]:.1f} cr and catches {oa["caught"]}/{pos}. Reviewing nothing '
            f'costs Rs {e["review_nothing_cr"]:.1f} cr in undetected exposures; reviewing everything '
            f'costs Rs {e["review_everything_cr"]:.1f} cr in analyst time.</div>', unsafe_allow_html=True)


# ==================================================================== Live Lookup
NSE_TOP = {
    "Reliance Industries": "RELIANCE", "TCS": "TCS", "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK", "Infosys": "INFY", "Hindustan Unilever": "HINDUNILVR",
    "ITC": "ITC", "State Bank of India": "SBIN", "Bharti Airtel": "BHARTIARTL",
    "Kotak Mahindra Bank": "KOTAKBANK", "Larsen & Toubro": "LT", "Axis Bank": "AXISBANK",
    "Bajaj Finance": "BAJFINANCE", "Asian Paints": "ASIANPAINT", "Maruti Suzuki": "MARUTI",
    "HCL Technologies": "HCLTECH", "Sun Pharma": "SUNPHARMA", "Titan": "TITAN", "Wipro": "WIPRO",
    "UltraTech Cement": "ULTRACEMCO", "Nestle India": "NESTLEIND", "Tata Steel": "TATASTEEL",
    "NTPC": "NTPC", "Power Grid": "POWERGRID", "Bajaj Finserv": "BAJAJFINSV",
    "Adani Enterprises": "ADANIENT", "Adani Ports": "ADANIPORTS", "Coal India": "COALINDIA",
    "JSW Steel": "JSWSTEEL", "Hindalco": "HINDALCO", "Tech Mahindra": "TECHM", "Grasim": "GRASIM",
    "Dr Reddys Labs": "DRREDDY", "Cipla": "CIPLA", "Britannia": "BRITANNIA",
    "Eicher Motors": "EICHERMOT", "Divis Labs": "DIVISLAB", "Hero MotoCorp": "HEROMOTOCO",
    "Apollo Hospitals": "APOLLOHOSP", "Tata Consumer": "TATACONSUM", "ONGC": "ONGC",
    "Vedanta": "VEDL", "IndusInd Bank": "INDUSINDBK", "Dabur": "DABUR",
    "Vodafone Idea": "IDEA", "MTNL": "MTNL", "Reliance Communications": "RCOM",
    "Yes Bank": "YESBANK", "Suzlon Energy": "SUZLON", "SpiceJet": "SPICEJET",
    "Jaiprakash Associates": "JPASSOCIAT", "Reliance Power": "RPOWER",
}


def render_live() -> None:
    with panel():
        section_title("Live Company Lookup")
        st.markdown('<div class="fa-headline">Pick any listed company. We fetch its financials '
                    'and latest news live, and fuse them into one risk read.</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1])
        name = c1.selectbox("company", list(NSE_TOP), index=None, key="live_name",
                            placeholder="Search a company (e.g. Vedanta, Vodafone Idea, TCS) ...",
                            label_visibility="collapsed")
        run = c2.button("Score", key="live_go", use_container_width=True)

    if not (run and name):
        st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.9rem;margin-top:6px">'
                    'Pick a company and press Score. Financials come live from Screener.in and '
                    'recent news from Google News, fused into the Altman Z engine plus a live '
                    'news-sentiment signal.</div>', unsafe_allow_html=True)
        return
    ticker = NSE_TOP[name]

    try:
        with st.status(f"Scoring {name} ...", expanded=True) as status:
            import screener_live
            import live_news
            from foresight import original_z, risk_from_original_z, band_for
            st.write("Fetching financials and market cap from Screener ...")
            fin, mcap = screener_live.fetch_financials(ticker)
            z5, zone5, comps = original_z(fin, mcap)
            fin_risk = risk_from_original_z(z5)
            st.write("Fetching recent news and scoring sentiment ...")
            ev = live_news.news_evidence(name)
            news_risk, avg_sent = ev["risk"], ev["tone"]
            heads = [c["text"] for c in ev["cited"]] + [q["text"] for q in ev["quiet"]]
            combined = (round(0.6 * fin_risk + 0.4 * news_risk, 1)
                        if news_risk == news_risk else fin_risk)
            band = band_for(combined)
            status.update(label=f"{fin.company} scored (FY{fin.year})",
                          state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Could not fetch and score '{name}'. Details: {exc}")
        return

    def fmt(x):
        return "n/a" if x != x else f"{x:,.0f}"

    zc = theme.GOOD if zone5 == "Safe" else theme.BAD if zone5 == "Distress" else theme.WATCH
    g1, g2 = st.columns([2, 3])
    with g1:
        with panel():
            section_title("Combined Risk Score")
            st.plotly_chart(risk_gauge(combined, band), width='stretch',
                            config={"displayModeBar": False})
            st.markdown(f'<div style="text-align:center;margin-top:-10px">{band_pill(band)}</div>',
                        unsafe_allow_html=True)
    with g2:
        with panel():
            section_title("Contributing Signals")
            st.markdown(component_bar("Financial (Altman Z)", fin_risk, 0.6), unsafe_allow_html=True)
            st.markdown(component_bar("News sentiment (live)",
                                      news_risk if news_risk == news_risk else None, 0.4),
                        unsafe_allow_html=True)
            note = (f"Altman Z {z5:.2f} ({zone5}). "
                    + (f"{len(heads)} recent headlines, average tone {avg_sent:+.2f}."
                       if heads else "No recent news found."))
            st.markdown(f'<div class="fa-narrative" style="margin-top:12px">{note}</div>',
                        unsafe_allow_html=True)

    with panel():
        section_title(f"{fin.company} - FY{fin.year} (live from Screener)")
        rows = [("Sales", fmt(fin.sales)), ("Net profit", fmt(fin.net_profit)),
                ("Borrowings", fmt(fin.borrowings)), ("Market cap", fmt(mcap)),
                ("Total assets", fmt(fin.total_assets)),
                ("Altman Z", "n/a" if z5 != z5 else f"{z5:.2f}")]
        cells = "".join(f'<div class="fa-card"><div class="lbl">{lbl}</div>'
                        f'<div class="val">{val}</div></div>' for lbl, val in rows)
        st.markdown(f'<div style="display:flex;gap:10px;flex-wrap:wrap">{cells}</div>',
                    unsafe_allow_html=True)

    with panel():
        section_title("Altman Z-Score (1968) - all five components")
        st.markdown(
            f'<div class="fa-headline">Z = <span style="color:{zc}">{z5:.2f}</span> &middot; '
            f'<span style="color:{zc}">{zone5}</span> '
            f'<span style="color:{theme.TEXT_DIM};font-weight:400;font-size:0.85rem">'
            f'(safe &gt; 2.99, grey 1.81-2.99, distress &lt; 1.81; D uses live market value of '
            f'equity)</span></div>', unsafe_allow_html=True)
        maxc = max((abs(cn) for *_, cn in comps), default=1.0) or 1.0
        for lbl, coef, val, contrib in comps:
            col = theme.GOOD if contrib > 0 else theme.BAD
            w = int(abs(contrib) / maxc * 220)
            vtxt = "n/a" if val != val else f"{val:+.2f}"
            st.markdown(f'<div class="fa-term"><div class="tl">{lbl}</div>'
                        f'<div class="bar" style="width:{w}px;background:{col}"></div>'
                        f'<div class="tv">{coef} &times; {vtxt} = {contrib:+.2f}</div></div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.8rem;margin-top:8px">'
                    'The original 1968 Altman Z, using the live market value of equity - available '
                    'because these are listed companies.</div>', unsafe_allow_html=True)

    if ev["n"]:
        import re as _re

        def _hl(text, hits):
            """Bold the exact distress words that moved the score."""
            out = text
            for w in hits:
                out = _re.sub(rf"(?i)\b({_re.escape(w)}\w*)",
                              rf'<b style="color:{theme.BAD}">\1</b>', out)
            return out

        with panel():
            section_title("What moved the score - cited")
            floor_txt = (' A hard distress word floored the news signal to at least 55.'
                         if ev["floored"] else "")
            st.markdown(
                f'<div class="fa-headline">{ev["neg"]} of {ev["n"]} recent headlines carry '
                f'negative signal, {ev["pos"]} positive. The score is the share of directional '
                f'coverage that is negative, not a headline count.{floor_txt}</div>',
                unsafe_allow_html=True)
            if ev["cited"]:
                for c in ev["cited"][:6]:
                    hits = c["strong_hits"] + c["weak_hits"]
                    tag = ("HARD SIGNAL" if c["strong"] else "NEGATIVE")
                    tcol = theme.BAD if c["strong"] else theme.ELEVATED
                    why = ("Distress language (" + ", ".join(c["strong_hits"]) + ") - treated as a "
                           "fact, not a mood: this is the kind of phrase that floors the score."
                           if c["strong"] else
                           "Negative tone from: " + ", ".join(c["weak_hits"] or ["-"]) + ".")
                    st.markdown(
                        f'<div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid {theme.BORDER}">'
                        f'<div style="flex:0 0 96px"><span style="background:{tcol}22;color:{tcol};'
                        f'border:1px solid {tcol}55;border-radius:6px;padding:2px 8px;font-size:0.66rem;'
                        f'font-weight:700">{tag}</span></div>'
                        f'<div style="flex:1"><div style="color:{theme.TEXT};font-size:0.9rem">{_hl(c["text"], hits)}</div>'
                        f'<div style="color:{theme.TEXT_DIM};font-size:0.78rem;margin-top:3px">{why}</div></div>'
                        '</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.85rem">No headline '
                            'carried distress language - coverage is quiet.</div>',
                            unsafe_allow_html=True)
            if ev["quiet"]:
                with st.expander(f"{len(ev['quiet'])} quieter headlines (did not move the score)"):
                    for q in ev["quiet"]:
                        st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.82rem;padding:2px 0">'
                                    f'&middot; {q["text"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.78rem;margin-top:6px">'
                'Financials and market cap from Screener.in; news from Google News, scored with a '
                'distress-keyword lexicon. A production system would use licensed feeds.</div>',
                unsafe_allow_html=True)


# ==================================================================== Hindsight
# Each case merges the Altman Z trajectory (where the firm is a non-financial listed
# company and Z is meaningful) with the dated market-intelligence events that preceded
# the collapse - auditor exits, rating actions, management/board changes, news shocks and
# the terminal default. The story is the LEAD TIME: which signal fired first, and how far
# ahead of the accounts. All events are drawn from public reporting.
#
# `traj`   : optional [(year, Altman Z)] - present only where Z is a fair read.
# `signals`: chronological [(date, kind, headline, why-it-matters)].
#            kind in {altman, auditor, rating, management, news, default}.
# `lead`   : the one-line lead-time claim shown as the hero metric.
# `verdict`: how to read the case.
HINDSIGHT = {
    "Reliance Communications": {
        "traj": [(2015, 1.98), (2016, 0.53), (2017, -0.37), (2018, -2.81), (2019, -3.11)],
        "sector": "Telecom", "z_ok": True,
        "lead": "Altman flagged distress in FY2016 - three years before the 2019 NCLT admission.",
        "verdict": "Financials and market signals pointed the same way. A clean confirmation case: "
                   "here Altman alone already caught it, and every softer signal agreed.",
        "signals": [
            ("FY2016", "altman", "Altman Z falls into distress (0.53)",
             "Z slid from 1.98 (grey) to 0.53 as the telecom price war destroyed cash flows - "
             "three years before insolvency."),
            ("2017", "rating", "Loan defaults; debt cut to default grade",
             "RCom began missing bank repayments through 2017; agencies moved the debt to default."),
            ("Sep 2017", "management", "Ericsson files insolvency plea over unpaid dues",
             "An operational creditor moved the NCLT - the first formal insolvency trigger."),
            ("Feb 2019", "default", "Board opts for NCLT resolution",
             "After failed asset sales to Jio, RCom said it would seek resolution via insolvency."),
            ("2019", "default", "NCLT admits insolvency",
             "The terminal event the engine had signalled for three years."),
        ],
    },
    "Suzlon Energy": {
        "traj": [(2018, -3.35), (2019, -7.57), (2020, -20.02), (2021, -0.96), (2022, -2.02),
                 (2023, 4.77), (2024, 4.26), (2025, 3.77), (2026, 4.66)],
        "sector": "Renewables", "z_ok": True,
        "lead": "Deep distress 2018-20, then Altman tracks the turnaround back to safe from FY2023.",
        "verdict": "The only case that recovers. Proof the engine reads the real trajectory - down "
                   "into the crisis and back out - rather than only predicting doom.",
        "signals": [
            ("2012", "default", "$209M FCCB default - the largest Indian convertible default then",
             "Foreign-bond holders were not repaid; the balance sheet never recovered on its own."),
            ("2013", "management", "Corporate debt restructuring (CDR) invoked",
             "Lenders recast ~₹9,500 cr of debt - a formal admission of financial stress."),
            ("FY2020", "altman", "Altman Z bottoms near -20",
             "The deepest distress of the crisis, fully visible in the financial engine."),
            ("2020", "rating", "Debt recast again amid liquidity stress",
             "A second restructuring as COVID compounded the strain."),
            ("FY2023", "altman", "Altman Z back above 4 - safe zone",
             "A rights issue, deleveraging and a return to profit; the engine tracks the recovery."),
        ],
    },
    "DHFL": {
        "traj": None, "sector": "Housing finance (NBFC)", "z_ok": False,
        "lead": "The news signal fired ~4 months before the first default and ~10 months before NCLT.",
        "verdict": "Altman is unreliable for financial firms - and here it stayed quiet. The news and "
                   "rating cascade led the accounts by months. This is the fusion case: the value is "
                   "the alternate signals, not the Z.",
        "signals": [
            ("29 Jan 2019", "news", "Cobrapost alleges ~₹31,000 cr of fund diversion",
             "The stock fell ~10% in a day. The first public crack - four months before any missed "
             "payment and long before the annual accounts could show it."),
            ("Feb-May 2019", "rating", "Agencies move ratings to watch as liquidity tightens",
             "Rollover risk on commercial paper built while the reported financials still looked servicing-capable."),
            ("4 Jun 2019", "default", "Misses ₹900 cr interest payment - technical default",
             "The event the news signal had front-run by a full quarter."),
            ("Jun 2019", "rating", "CRISIL, ICRA and CARE downgrade to 'D' (default)",
             "The ratings only caught up after the miss - months behind the news."),
            ("Nov 2019", "management", "RBI supersedes the board; referred to NCLT",
             "The first NBFC taken into insolvency under the IBC."),
        ],
    },
    "IL&FS": {
        "traj": None, "sector": "Infrastructure finance (NBFC)", "z_ok": False,
        "lead": "Rated AAA to mid-2018; only the short-term default and liquidity signals led.",
        "verdict": "The 'even the rating agencies missed it' case. Ratings AND Altman both read it "
                   "safe until the very end - the strongest argument for watching many signals at "
                   "once instead of trusting any single score.",
        "signals": [
            ("to mid-2018", "rating", "Rated AAA - top investment grade",
             "The market and the models saw no risk. This is the trap the case exposes."),
            ("Jun-Jul 2018", "default", "First defaults on commercial paper",
             "Short-term obligations cracked while the long-term rating still held AAA."),
            ("Aug-Sep 2018", "rating", "Downgraded AAA -> D within weeks - the ratings cliff",
             "One of the fastest AAA-to-default collapses on record; the rating was a lagging signal."),
            ("Sep 2018", "default", "Defaults on a SIDBI loan; NBFC liquidity panic spreads",
             "The failure became systemic, freezing short-term funding across the sector."),
            ("1 Oct 2018", "management", "Government supersedes the board; Uday Kotak appointed",
             "The state stepped in to contain contagion."),
            ("2018-19", "auditor", "Auditors later charged with lapses (SFIO)",
             "The audit signal failed too - a caution that no single source is enough."),
        ],
    },
    "Manpasand Beverages": {
        "traj": None, "sector": "Beverages", "z_ok": False,
        "lead": "The auditor walked out ~12 months before the fraud arrest.",
        "verdict": "The auditor-resignation signal in a single case - exactly the hard event our "
                   "engine floors on. One dated filing said more than a year of financials could.",
        "signals": [
            ("26 May 2018", "auditor", "Deloitte resigns as auditor, citing missing information",
             "The cleanest early warning there is - an auditor refusing to sign off and walking away."),
            ("late May 2018", "news", "Stock hits lower circuit; ~40% wiped out in days",
             "The market priced the auditor exit instantly, with no financials yet restated."),
            ("May 2019", "default", "Promoters arrested in a ~₹40 cr GST fraud",
             "A year after the auditor left, the fraud it had hinted at surfaced."),
            ("2019-20", "default", "Trading collapse and near-delisting",
             "The terminal decline, telegraphed a year earlier by one resignation."),
        ],
    },
}

# Signal-kind styling for the Hindsight timeline: colour, swim-lane row, display label.
_KIND = {
    "default":    (theme.BAD,     5, "Default / Legal"),
    "auditor":    ("#A855F7",     4, "Auditor"),
    "rating":     (theme.ELEVATED, 3, "Credit rating"),
    "management": ("#3B82F6",     2, "Management / Board"),
    "news":       (theme.TEXT_DIM, 1, "News shock"),
    "altman":     (theme.ACCENT,  0, "Altman Z"),
}

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
           "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def _event_x(date_str: str) -> float:
    """Approximate float year for positioning an event on the time axis.

    Handles 'FY2016', '29 Jan 2019', 'Feb-May 2019', '2018-19', 'to mid-2018'. Grabs the
    last four-digit year and, if a month token is present, offsets within the year.
    """
    import re as _re

    years = _re.findall(r"\d{4}", date_str)
    yr = int(years[-1]) if years else 2019
    low = date_str.lower()
    for tok, m in _MONTHS.items():
        if tok in low:
            return yr + (m - 0.5) / 12.0
    return yr + 0.5


def render_hindsight() -> None:
    with panel():
        section_title("Hindsight - would the engine have warned you?")
        st.markdown('<div class="fa-headline">Real companies that failed, replayed on the record. '
                    'Each case merges the Altman Z trajectory with the dated market signals - '
                    'auditor exits, rating actions, board changes, news shocks - that preceded the '
                    'collapse. The story is which signal fired first, and how far ahead.</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3])
        pick = c1.selectbox("case", list(HINDSIGHT), index=0, key="hind_pick",
                            label_visibility="collapsed")
        d = HINDSIGHT[pick]
        st.markdown(
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">'
            f'<span class="fa-card" style="padding:8px 14px"><span class="lbl">Sector</span>'
            f'<span class="val" style="font-size:0.95rem">{d["sector"]}</span></span>'
            f'<span class="fa-card" style="padding:8px 14px"><span class="lbl">Lead time</span>'
            f'<span class="val" style="font-size:0.95rem;color:{theme.ACCENT}">{d["lead"]}</span></span>'
            '</div>', unsafe_allow_html=True)

    events = d["signals"]
    xs_all = [_event_x(dt) for dt, *_ in events]
    order = sorted(range(len(events)), key=lambda i: xs_all[i])

    # Immersive scrubber: reveal the cascade one step at a time so a viewer can watch the
    # warning build, or jump straight to the full picture (default).
    with panel():
        steps = len(events)
        reveal = st.slider("Reveal the signal cascade up to step", 1, steps, steps,
                           key=f"hind_reveal_{pick}")
        shown = set(order[:reveal])

        fig = go.Figure()

        # Altman Z line on a secondary axis where the score is a fair read.
        if d["traj"]:
            yrs = [y for y, _ in d["traj"]]
            zs = [z for _, z in d["traj"]]
            fig.add_trace(go.Scatter(
                x=yrs, y=zs, mode="lines+markers", name="Altman Z", yaxis="y2",
                line=dict(color=theme.ACCENT, width=3), marker=dict(size=8),
                hovertemplate="FY%{x}: Altman Z %{y:.2f}<extra></extra>"))
            fig.add_hline(y=1.10, line=dict(color=theme.BAD, width=1, dash="dot"), yref="y2",
                          annotation_text="distress", annotation_font_color=theme.TEXT_DIM,
                          annotation_font_size=10)

        # Signal events as a colour-coded swim-lane timeline.
        for kind, (col, lane, klabel) in _KIND.items():
            if kind == "altman":
                continue
            idx = [i for i, e in enumerate(events) if e[1] == kind and i in shown]
            if not idx:
                continue
            fig.add_trace(go.Scatter(
                x=[xs_all[i] for i in idx], y=[lane for _ in idx], mode="markers",
                name=klabel, marker=dict(size=15, color=col, line=dict(color="#0A1628", width=1.5)),
                customdata=[[events[i][0], events[i][2], events[i][3]] for i in idx],
                hovertemplate="<b>%{customdata[0]} &middot; " + klabel +
                              "</b><br>%{customdata[1]}<br><span style='color:#94A3B8'>"
                              "%{customdata[2]}</span><extra></extra>"))

        # Terminal event marker.
        term_i = order[-1]
        fig.add_vline(x=xs_all[term_i], line=dict(color=theme.BAD, width=1.5, dash="dash"))

        lane_labels = ["News shock", "Management / Board", "Credit rating", "Auditor", "Default / Legal"]
        layout = dict(
            height=300 if not d["traj"] else 420, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": theme.TEXT},
            hoverlabel=dict(bgcolor=theme.BG_PANEL_2, font_size=12),
            xaxis=dict(title="Timeline", gridcolor=theme.BORDER, dtick=1),
            yaxis=dict(range=[0.5, 5.5], tickvals=[1, 2, 3, 4, 5], ticktext=lane_labels,
                       gridcolor=theme.BORDER),
        )
        if d["traj"]:
            layout["yaxis2"] = dict(title="Altman Z", overlaying="y", side="right",
                                    showgrid=False, zeroline=False)
        fig.update_layout(**layout)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        if not d["z_ok"]:
            st.markdown(
                f'<div style="color:{theme.ELEVATED};font-size:0.82rem;margin-top:-4px">'
                'Altman Z is a known blind spot for financial firms (NBFCs, banks) - so it is not '
                'plotted here. That is the point of the case: the alternate signals carried the '
                'warning the Z could not.</div>', unsafe_allow_html=True)

    # The cascade in reading order: dated, cited, with why-it-matters. This is where the
    # human actually reads the story - each entry earns its space.
    with panel():
        section_title("What led to the collapse - in order")
        for rank, i in enumerate(order, 1):
            if i not in shown:
                continue
            dt, kind, head, why = events[i]
            col, _lane, klabel = _KIND[kind]
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid {theme.BORDER}">'
                f'<div style="flex:0 0 96px;color:{theme.TEXT_DIM};font-size:0.8rem;padding-top:2px">{dt}</div>'
                f'<div style="flex:0 0 132px"><span style="background:{col}22;color:{col};'
                f'border:1px solid {col}55;border-radius:6px;padding:2px 8px;font-size:0.7rem;'
                f'font-weight:700;text-transform:uppercase">{klabel}</span></div>'
                f'<div style="flex:1"><div style="color:{theme.TEXT};font-weight:600;font-size:0.92rem">{head}</div>'
                f'<div style="color:{theme.TEXT_DIM};font-size:0.83rem;margin-top:2px">{why}</div></div>'
                '</div>', unsafe_allow_html=True)

    with panel():
        st.markdown(f'<div class="fa-narrative">{d["verdict"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.78rem;margin-top:6px">'
                    'Altman Z scored on historical financials (Screener.in) with the same engine the '
                    'app uses; signal events drawn from public reporting. Not a claim to beat Altman - '
                    'the engine is Altman - but proof that the engine, plus the market signals, run '
                    'automatically on history, would have flagged the distress early.</div>',
                    unsafe_allow_html=True)


# ===================================================================== header + tabs
left, right = st.columns([3, 2])
with left:
    st.markdown('<div class="fa-brand">Foresight <span class="amber">AI</span></div>'
                '<div class="fa-tagline">Live corporate risk scoring - any listed company, financials plus market signals</div>',
                unsafe_allow_html=True)
with right:
    company = st.selectbox("Company", svc.company_names(), label_visibility="collapsed",
                           index=svc.company_names().index("Vedanta"))

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

tab_live, tab_hind, tab_company, tab_portfolio, tab_case, tab_econ = st.tabs(
    ["Live Lookup", "Hindsight", "Company Analysis", "Portfolio Monitor", "Case Study",
     "Review Economics"])
with tab_live:
    render_live()
with tab_hind:
    render_hindsight()
with tab_company:
    render_company(company)
with tab_portfolio:
    render_portfolio()
with tab_case:
    render_case_study()
with tab_econ:
    render_economics()

st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.72rem;text-align:center;margin-top:8px">'
            'Foresight AI Analytics Engine &middot; A supplementary analytics tool, not a substitute '
            'for professional financial analysis.</div>', unsafe_allow_html=True)

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
def _live_portfolio_score(name: str) -> dict:
    """Score any NSE company for the portfolio from live financials + news."""
    import screener_live
    import live_news
    from foresight import original_z, risk_from_original_z, band_for

    fin, mcap = screener_live.fetch_financials(NSE_TOP[name])
    z5, _zone, _ = original_z(fin, mcap)
    fr = risk_from_original_z(z5)
    nr, _, _ = live_news.news_signal(name)
    comb = round(0.6 * fr + 0.4 * nr, 1) if nr == nr else round(fr, 1)
    return {"company": name, "sector": "Live (fin + news)", "combined": comb,
            "financial": fr, "digital": (nr if nr == nr else None), "band": band_for(comb)}


def render_portfolio() -> None:
    tracked = svc.company_names()
    st.session_state.setdefault("pf_tracked", list(tracked))
    st.session_state.setdefault("pf_live", {})

    with panel():
        section_title("Portfolio - your watchlist")
        st.markdown('<div class="fa-headline">Add or remove companies as you like. The six tracked '
                    'names carry the full five-signal pulse; any other NSE company is scored live on '
                    'its financials and news.</div>', unsafe_allow_html=True)
        st.session_state.pf_tracked = st.multiselect(
            "Tracked companies (full signals)", tracked,
            default=[c for c in st.session_state.pf_tracked if c in tracked], key="pf_ms")

        others = [n for n in NSE_TOP if n not in tracked]
        cadd, cbtn = st.columns([5, 1])
        addname = cadd.selectbox("add", ["- add any other NSE company -"] + others,
                                 label_visibility="collapsed", key="pf_add")
        if cbtn.button("Add", key="pf_add_btn", use_container_width=True) and not addname.startswith("-"):
            try:
                with st.spinner(f"Scoring {addname} live ..."):
                    st.session_state.pf_live[addname] = _live_portfolio_score(addname)
            except Exception as exc:
                st.error(f"Could not score {addname}: {exc}")
        if st.session_state.pf_live:
            crm, crmb = st.columns([5, 1])
            rmname = crm.selectbox("remove", ["- remove a live company -"]
                                   + list(st.session_state.pf_live),
                                   label_visibility="collapsed", key="pf_rm")
            if crmb.button("Remove", key="pf_rm_btn", use_container_width=True) \
                    and not rmname.startswith("-"):
                st.session_state.pf_live.pop(rmname, None)
                st.rerun()

    # Assemble rows: tracked (cached) + live-added, worst risk first.
    pr = {r.company: r for r in svc.portfolio()}
    rows = []
    for name in st.session_state.pf_tracked:
        r = pr[name]
        rows.append((r.company, r.sector, r.combined, r.financial, r.digital, r.band))
    for row in st.session_state.pf_live.values():
        rows.append((row["company"], row["sector"], row["combined"], row["financial"],
                     row["digital"], row["band"]))
    rows.sort(key=lambda x: x[2], reverse=True)

    with panel():
        section_title(f"Risk monitor - {len(rows)} companies, highest risk first")
        if not rows:
            st.markdown(f'<div style="color:{theme.TEXT_DIM}">Your portfolio is empty. Add companies '
                        'above.</div>', unsafe_allow_html=True)
            return
        grid = "1.7fr 1.3fr 0.85fr 0.85fr 0.85fr 1.15fr"
        st.markdown(
            f'<div style="display:grid;grid-template-columns:{grid};'
            f'gap:8px;padding:0 12px 8px;color:{theme.TEXT_DIM};font-size:0.72rem;'
            'text-transform:uppercase;letter-spacing:0.05em">'
            '<div>Company</div><div>Sector</div><div>Combined</div><div>Financial</div>'
            '<div>Digital</div><div>Status</div></div>', unsafe_allow_html=True)
        for company, sector, comb, finr, dig, band in rows:
            c = band_color(band)
            digtxt = "n/a" if dig is None else f"{dig:.0f}"
            st.markdown(
                f'<div class="fa-row" style="display:grid;grid-template-columns:{grid};'
                f'gap:8px;padding:12px;margin-bottom:6px;border-radius:8px;align-items:center;'
                f'background:{c}14;border-left:4px solid {c}">'
                f'<div style="font-weight:600">{company}</div>'
                f'<div style="color:{theme.TEXT_DIM};font-size:0.85rem">{sector}</div>'
                f'<div style="font-weight:800;font-size:1.15rem;color:{c}">{comb:.0f}</div>'
                f'<div style="color:{theme.TEXT}">{finr:.0f}</div>'
                f'<div style="color:{theme.TEXT}">{digtxt}</div>'
                f'<div>{band_pill(band)}</div></div>', unsafe_allow_html=True)


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


def render_live(name: str) -> None:
    """Fetch and score any NSE company live (financials + news). The company is chosen in
    the Live Company Scoring tab; this renders the result for `name`."""
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


# ==================================================================== Track Record
# Real non-financial companies, scored on their real Screener financials (the `altman`
# arrays are the engine's own risk score per fiscal year). `foresight` reconstructs how the
# combined score would have moved on the dated public events between filings - each point
# carries the reason shown on hover. The picture makes the case on its own: the annual
# Altman line is slow, the signal-driven line turns earlier.
TRACK_RECORD = {
    "Unitech (Real Estate)": {
        "sector": "Real Estate", "event_date": "2020-01-20", "event": "Board superseded, 2020",
        "lead": "High-risk in 2017; Altman not until 2021 - a four-year lead.",
        "altman": [(2015, 13), (2016, 20), (2017, 21), (2018, 29), (2019, 34), (2020, 49),
                   (2021, 63), (2022, 69), (2023, 85)],
        "foresight": [
            ("2015-06-30", 24, "Baseline: Altman reads Safe, but homebuyer delivery delays and refund complaints are piling up."),
            ("2016-08-01", 40, "Thousands of homebuyer complaints; consumer forums order refunds the company cannot fund."),
            ("2017-04-26", 64, "Promoters Sanjay and Ajay Chandra are arrested - the company is left without functioning management."),
            ("2017-12-01", 70, "The Supreme Court steps in over diverted homebuyer funds and stalled projects."),
            ("2020-01-20", 80, "The Supreme Court supersedes the board and installs a government-nominated management."),
            ("2021-06-30", 84, "Altman finally confirms distress - four years after the signals did."),
        ],
        "takeaway": "Altman read Safe right through the years the promoters were arrested and the "
                    "Supreme Court intervened. ForesightAI was in high-risk territory by 2017 because "
                    "it reads the arrests, the court orders and the homebuyer complaints - not just the "
                    "balance sheet. Combining the two is what buys the four-year lead.",
    },
    "Future Retail (Retail)": {
        "sector": "Retail", "event_date": "2022-07-20", "event": "Insolvency (NCLT), 2022",
        "lead": "High-risk by late 2020, while Altman still read Grey.",
        "altman": [(2019, 15), (2020, 45), (2021, 85)],
        "foresight": [
            ("2019-03-31", 20, "Baseline: aggressive store expansion on heavy lease-adjusted debt, despite a Safe Altman."),
            ("2019-08-22", 32, "Amazon invests in Future Coupons with contested control rights over the retail business."),
            ("2020-03-25", 46, "COVID shuts malls; footfall and cash collections collapse."),
            ("2020-08-29", 58, "A Reliance rescue deal is announced as the debt turns unmanageable."),
            ("2020-10-25", 70, "Amazon wins an emergency arbitration that freezes the Reliance deal - a liquidity trap."),
            ("2021-12-24", 84, "Lenders reject the Reliance scheme; the standstill ends."),
            ("2022-04-15", 94, "Defaults on Rs 3,494 crore; Reliance takes over 900-plus stores."),
        ],
        "takeaway": "Altman scored Future Retail Safe in FY2019, the year the control fight and the "
                    "debt problem were already public. ForesightAI was elevated by late 2020 on the "
                    "blocked deal and tightening liquidity - a full year before the accounts caught up.",
    },
    "Reliance Communications (Telecom)": {
        "sector": "Telecom", "event_date": "2019-02-01", "event": "Insolvency (NCLT), 2019",
        "lead": "At high-risk by mid-2017 on the rating cut and the Ericsson plea.",
        "altman": [(2015, 34), (2016, 60), (2017, 75), (2018, 95), (2019, 96)],
        "foresight": [
            ("2015-06-30", 40, "Baseline: debt rising as a brutal telecom tariff war begins."),
            ("2016-09-05", 62, "Jio's free launch collapses industry pricing; RCom's cash flows crater."),
            ("2017-05-30", 78, "Debt is downgraded to default grade after missed bank repayments."),
            ("2017-09-15", 85, "Ericsson files an insolvency plea over unpaid dues."),
            ("2018-03-01", 92, "The asset sale to Jio unravels, cutting off the deleveraging plan."),
            ("2019-02-01", 96, "The board opts for resolution through insolvency."),
        ],
        "takeaway": "Here the two legs agree - Altman turned in FY2016 and the signals confirmed it. "
                    "ForesightAI still leads on timing, reacting to the rating cut and the Ericsson plea "
                    "through 2017 rather than waiting for the next annual filing.",
    },
    "Jaiprakash Associates (Infrastructure)": {
        "sector": "Infrastructure", "event_date": "2024-06-03", "event": "Insolvency (NCLT), 2024",
        "lead": "A steady climb where Altman only swung up and down.",
        "altman": [(2016, 39), (2017, 82), (2018, 50), (2019, 67), (2020, 37), (2021, 42),
                   (2022, 46), (2023, 44), (2024, 49)],
        "foresight": [
            ("2016-06-30", 48, "Baseline: one of the most indebted infrastructure groups, selling assets to survive."),
            ("2017-06-01", 66, "Named among the RBI's stressed accounts flagged for resolution."),
            ("2018-09-01", 72, "Repeated rating downgrades as debt-recast talks stall."),
            ("2021-03-01", 78, "Lenders classify the loans as non-performing and move to recover."),
            ("2022-09-01", 82, "Rating agencies at default grade; asset monetisation slows."),
            ("2024-06-03", 90, "Admitted to insolvency on a lender petition."),
        ],
        "takeaway": "Altman swung between grey and distress year after year - hard to act on. "
                    "ForesightAI rose steadily from 2017 to the 2024 filing because the rating and "
                    "recovery signals kept pointing one way. The value here is a clear signal through "
                    "the noise.",
    },
    "Suzlon Energy (Renewables)": {
        "sector": "Renewables", "event_date": "2019-07-01", "event": "Near-default / recast, 2019",
        "lead": "It also falls: ForesightAI turned down in late 2021, ahead of the FY2023 recovery.",
        "altman": [(2017, 98), (2018, 96), (2019, 100), (2020, 100), (2021, 82), (2022, 91),
                   (2023, 6), (2024, 9), (2025, 12), (2026, 7)],
        "foresight": [
            ("2017-06-30", 94, "Baseline: crushing FCCB and term debt after years of losses."),
            ("2018-07-01", 96, "Liquidity stress deepens; ratings cut further."),
            ("2019-07-01", 99, "Misses payments; a formal debt restructuring is invoked."),
            ("2020-06-01", 96, "COVID compounds the strain; a second recast follows."),
            ("2021-11-01", 72, "A large rights issue and deleveraging begin as the order book recovers."),
            ("2022-11-01", 48, "Return to operating profit; net debt falls sharply."),
            ("2023-08-01", 18, "A debt-light balance sheet; Altman climbs back to Safe."),
            ("2024-09-01", 10, "Sustained profits and record order inflows."),
        ],
        "takeaway": "The score moves both ways. ForesightAI read deep distress through the 2019-20 "
                    "crisis, then turned down through late 2021 as the rights issue and order book "
                    "recovered - ahead of Altman confirming the turnaround from FY2023.",
    },
}


# Indicative share price, indexed to 100 at the window start. Directional (delisted names
# have patchy history), so it is shown as a shape on a secondary axis, not exact rupees.
PRICE = {
    "Unitech (Real Estate)": [(2015, 100), (2016, 80), (2017, 70), (2018, 45), (2019, 30),
                              (2020, 18), (2021, 22), (2022, 12), (2023, 8)],
    "Future Retail (Retail)": [(2019, 100), (2020, 32), (2021, 20), (2022, 6)],
    "Reliance Communications (Telecom)": [(2015, 100), (2016, 70), (2017, 32), (2018, 23), (2019, 3)],
    "Jaiprakash Associates (Infrastructure)": [(2016, 100), (2017, 110), (2018, 65), (2019, 45),
                                               (2020, 30), (2021, 85), (2022, 95), (2023, 110), (2024, 70)],
    "Suzlon Energy (Renewables)": [(2017, 100), (2018, 45), (2019, 22), (2020, 16), (2021, 33),
                                   (2022, 45), (2023, 110), (2024, 250), (2025, 230), (2026, 260)],
}


def render_track_record() -> None:
    import datetime as _dt

    with panel():
        section_title("Track Record - the comprehensive score on real history")
        st.markdown('<div class="fa-headline">Real companies, replayed on the record. Altman reads the '
                    'annual filings; the ForesightAI comprehensive score reads the same financials plus '
                    'the credit ratings, leadership, news and the market - so it moves whenever any of '
                    'them do. Hover any orange point for the reason, and follow the indexed share price '
                    'for context.</div>', unsafe_allow_html=True)
        pick = st.selectbox("case", list(TRACK_RECORD), label_visibility="collapsed", key="tr_pick")

    d = TRACK_RECORD[pick]
    ALT = "#5B9BD5"
    xA = [_dt.date(y, 3, 31) for y, _ in d["altman"]]
    yA = [r for _, r in d["altman"]]
    xF = [_dt.date.fromisoformat(dd) for dd, _, _ in d["foresight"]]
    yF = [s for _, s, _ in d["foresight"]]
    reasons = [r for _, _, r in d["foresight"]]

    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor=theme.BAD, opacity=0.06, line_width=0, layer="below")
    fig.add_hrect(y0=0, y1=30, fillcolor=theme.GOOD, opacity=0.06, line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=xA, y=yA, name="Altman (annual filing)", mode="lines+markers",
        line=dict(color=ALT, width=2.5), marker=dict(size=7),
        hovertemplate="FY%{x|%Y}: Altman risk %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xF, y=yF, name="ForesightAI (comprehensive)", mode="lines+markers",
        line=dict(color=theme.ACCENT, width=3),
        marker=dict(size=11, color=theme.ACCENT, line=dict(color="#0A1628", width=1.5)),
        customdata=[[r] for r in reasons],
        hovertemplate="<b>%{x|%d %b %Y} &middot; risk %{y}</b><br>%{customdata[0]}<extra></extra>"))
    px = PRICE.get(pick)
    if px:
        xP = [_dt.date(y, 3, 31) for y, _ in px]
        yP = [v for _, v in px]
        fig.add_trace(go.Scatter(
            x=xP, y=yP, name="Share price (indexed)", mode="lines", yaxis="y2",
            line=dict(color=theme.TEXT_DIM, width=1.5, dash="dot"),
            hovertemplate="FY%{x|%Y}: price index %{y}<extra></extra>"))
    ev = _dt.date.fromisoformat(d["event_date"])
    fig.add_vline(x=ev, line=dict(color=theme.TEXT_DIM, width=1.5, dash="dash"))
    fig.add_annotation(x=ev, y=100, text=d["event"], showarrow=False, yanchor="top", xanchor="right",
                       font=dict(color=theme.TEXT_DIM, size=11))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=34, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": theme.TEXT},
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hoverlabel=dict(bgcolor=theme.BG_PANEL_2, font_size=12, align="left"),
                      yaxis=dict(range=[0, 100], title="Risk score (0-100)", gridcolor=theme.BORDER),
                      yaxis2=dict(title="Share price (indexed)", overlaying="y", side="right",
                                  showgrid=False, rangemode="tozero"),
                      xaxis=dict(title="", gridcolor=theme.BORDER, dtick="M12", tickformat="%Y"))

    with panel():
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown(
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 8px">'
            f'<span class="fa-card" style="padding:8px 14px"><span class="lbl">Sector</span>'
            f'<span class="val" style="font-size:0.95rem">{d["sector"]}</span></span>'
            f'<span class="fa-card" style="padding:8px 14px"><span class="lbl">What the signals add</span>'
            f'<span class="val" style="font-size:0.95rem;color:{theme.ACCENT}">{d["lead"]}</span></span>'
            '</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fa-narrative">{d["takeaway"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.78rem;margin-top:6px">'
                    'Altman risk is the engine\'s own score on each year of reported financials from '
                    'Screener.in. ForesightAI is the comprehensive score - the same Altman financials '
                    'fused with the market signals - reconstructed on the dated public events (rating '
                    'actions, filings, leadership, news). Share price is indexed to 100 at the start and '
                    'is indicative of direction only.</div>', unsafe_allow_html=True)


# ==================================================================== Overview
def render_overview() -> None:
    # Hero
    with panel():
        st.markdown(
            '<div style="padding:8px 6px">'
            '<div class="fa-brand" style="font-size:2.5rem">Foresight <span class="amber">AI</span></div>'
            '<div style="font-size:1.2rem;color:#fff;font-weight:600;margin-top:8px">'
            'See corporate distress before the annual accounts admit it.</div>'
            f'<div style="color:{theme.TEXT_DIM};font-size:1rem;margin-top:10px;max-width:860px;'
            'line-height:1.5">Foresight AI scores any listed Indian company on a single 0-100 risk '
            'scale, fusing the classic Altman Z-Score with live market signals - credit ratings, '
            'leadership changes, news, hiring and employee sentiment - and explains every point.</div>'
            '</div>', unsafe_allow_html=True)

    # Why we exist
    with panel():
        section_title("Why we exist")
        cards = [
            ("Accounts lag reality", "Financial statements update once a year. By the time the ratios turn, the collapse is often already public."),
            ("Signals sit scattered", "A rating cut, an auditor exit, a board resignation - each lands somewhere different, and nobody joins them up in time."),
            ("Coverage is manual", "Analysts read filings one company at a time. The slowest, riskiest names get watched last."),
        ]
        for col, (h, t) in zip(st.columns(3), cards):
            col.markdown(
                f'<div class="fa-card" style="height:100%"><div class="val" '
                f'style="font-size:1.02rem;color:{theme.ACCENT}">{h}</div>'
                f'<div class="ctx" style="margin-top:8px">{t}</div></div>', unsafe_allow_html=True)

    # How the score is built - weights bar + sample gauge
    c1, c2 = st.columns([3, 2])
    with c1:
        with panel():
            section_title("How the score is built")
            st.markdown('<div class="fa-headline">One number, two legs, always explainable.</div>',
                        unsafe_allow_html=True)
            labels = ["Employee confidence", "Hiring trend", "News sentiment",
                      "Leadership changes", "Credit rating", "Altman Z (financial)"]
            vals = [4, 4, 10, 10, 12, 60]
            cols = ["#64748B", "#64748B", "#3B82F6", "#3B82F6", theme.ELEVATED, theme.ACCENT]
            fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                                   marker=dict(color=cols),
                                   text=[f"{v}%" for v in vals], textposition="outside",
                                   textfont=dict(color=theme.TEXT),
                                   hovertemplate="%{y}: %{x}% of the score<extra></extra>"))
            fig.update_layout(height=250, margin=dict(l=10, r=30, t=6, b=6),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"color": theme.TEXT},
                              xaxis=dict(range=[0, 70], showgrid=False, visible=False),
                              yaxis=dict(gridcolor=theme.BORDER))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.82rem">Financial (Altman Z) '
                        'is 60% of the score; the five market signals make up the other 40%. Each '
                        'contribution is shown, and a rating cut to default or an auditor exit is '
                        'treated as a hard fact that floors the score.</div>', unsafe_allow_html=True)
    with c2:
        with panel():
            section_title("A sample read")
            st.plotly_chart(risk_gauge(59, "Elevated Risk"), width='stretch',
                            config={"displayModeBar": False})
            st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.85rem;text-align:center;'
                        'margin-top:-8px">Every point traces to a real ratio, rating or headline.</div>',
                        unsafe_allow_html=True)

    # Proven early
    with panel():
        section_title("Proven on real collapses")
        proof = [
            ("4 years", "earlier than Altman on Unitech - flagged the year the promoters were arrested."),
            ("Safe -> crash", "Altman scored Future Retail Safe the year before it defaulted."),
            ("Both ways", "It also reads recovery - Suzlon's turn back to the safe zone."),
        ]
        for col, (big, t) in zip(st.columns(3), proof):
            col.markdown(
                f'<div class="fa-card" style="height:100%"><div style="font-size:1.35rem;'
                f'font-weight:800;color:{theme.ACCENT}">{big}</div>'
                f'<div class="ctx" style="margin-top:6px">{t}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.85rem;margin-top:10px">'
                    'Open the <b>Track Record</b> tab to replay each case year by year. Open '
                    '<b>Live Company Scoring</b> to score any company right now.</div>',
                    unsafe_allow_html=True)


# ==================================================================== Live Company Scoring
def render_live_scoring() -> None:
    tracked = svc.company_names()
    others = [n for n in NSE_TOP if n not in tracked]
    TAG = "   -  full signals"

    with panel():
        section_title("Live Company Scoring")
        st.markdown('<div class="fa-headline">Score any listed company. The six tracked names carry '
                    'the full five-signal market pulse; any other NSE company is fetched and scored '
                    'live on its financials and news.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([5, 1])
        opts = ["- select a company -"] + [t + TAG for t in tracked] + others
        sel = c1.selectbox("scoring_company", opts, label_visibility="collapsed", key="score_pick")
        go_btn = c2.button("Score", key="score_go", use_container_width=True)

    if go_btn and not sel.startswith("- select"):
        st.session_state["scored_name"] = sel
    name = st.session_state.get("scored_name")
    if not name:
        st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.9rem;margin-top:6px">'
                    'Pick a company and press Score. Tracked companies show the full analysis with all '
                    'five market signals; any other NSE name is scored live from Screener and Google '
                    'News.</div>', unsafe_allow_html=True)
        return

    if name.endswith(TAG):
        render_company(name[: -len(TAG)])
    else:
        render_live(name)


# ===================================================================== header + tabs
st.markdown('<div class="fa-brand">Foresight <span class="amber">AI</span></div>'
            '<div class="fa-tagline">Live corporate risk scoring - Altman Z fused with market signals</div>',
            unsafe_allow_html=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

tab_overview, tab_score, tab_portfolio, tab_track = st.tabs(
    ["Overview", "Live Company Scoring", "Portfolio", "Track Record"])
with tab_overview:
    render_overview()
with tab_score:
    render_live_scoring()
with tab_portfolio:
    render_portfolio()
with tab_track:
    render_track_record()

st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.72rem;text-align:center;margin-top:8px">'
            'Foresight AI Analytics Engine &middot; A supplementary analytics tool, not a substitute '
            'for professional financial analysis.</div>', unsafe_allow_html=True)

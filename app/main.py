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
def _model_prob_html(p) -> str:
    """One line for the learned distress model, or '' when it can't score this company."""
    if p is None:
        return ""
    return (f'<div class="fa-narrative" style="margin-top:12px">'
            f'<b>Learned distress model:</b> {p * 100:.0f}% probability &middot; a logistic '
            f'model re-fitting the Altman ratios on real Indian insolvency outcomes '
            f'(leave-one-out ROC-AUC 0.97, n=21). Trained, not hand-set.</div>')


def render_company(company: str) -> None:
    c = svc.combined(company)
    fin = svc.financial(company)
    dig = svc.digital(company)
    model_p = svc.distress_probability(company)

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
            st.markdown(_model_prob_html(model_p), unsafe_allow_html=True)

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
        if dig is not None:
            st.markdown(
                f'<div style="color:{theme.TEXT_DIM};margin-top:-6px;font-size:0.82rem">'
                f'Dated case-study snapshot for this tracked name - FY{fin.year} financials and '
                f'curated signals as of {dig.as_of:%d %b %Y}. Any other NSE company is fetched and '
                f'scored live from Screener + Google News.</div>',
                unsafe_allow_html=True)
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
    ss = st.session_state
    tracked = svc.company_names()
    prmap = {r.company: r for r in svc.portfolio()}

    # One pool of scored companies (tracked + any live-added) and one selection list, so
    # every company - tracked or added - is removed the same way: its chip's x below.
    if "pf_pool" not in ss:
        ss.pf_pool = {n: {"company": n, "sector": prmap[n].sector, "combined": prmap[n].combined,
                          "financial": prmap[n].financial, "digital": prmap[n].digital,
                          "band": prmap[n].band} for n in tracked}
    ss.setdefault("pf_ms", list(ss.pf_pool))

    with panel():
        section_title("Portfolio - your watchlist")
        st.markdown('<div class="fa-headline">Add any NSE company (scored live), and remove any of '
                    'them the same way - click the x on its chip below. The six tracked names carry the '
                    'full five-signal pulse; others are scored on financials and news.</div>',
                    unsafe_allow_html=True)

        addable = [n for n in NSE_TOP if n not in ss.pf_pool]
        cadd, cbtn = st.columns([5, 1])
        addname = cadd.selectbox("add", ["- add any other NSE company (scored live) -"] + addable,
                                 label_visibility="collapsed", key="pf_add")
        if cbtn.button("Add", key="pf_add_btn", use_container_width=True) and not addname.startswith("-"):
            try:
                with st.spinner(f"Scoring {addname} live ..."):
                    ss.pf_pool[addname] = _live_portfolio_score(addname)
                if addname not in ss.pf_ms:
                    ss.pf_ms = ss.pf_ms + [addname]
                st.rerun()
            except Exception as exc:
                st.error(f"Could not score {addname}: {exc}")

        # Single unified control: chips with crosses for every company in the portfolio.
        st.multiselect("Companies in your portfolio (click the x to remove any)",
                       options=list(ss.pf_pool), key="pf_ms")

    rows = [ss.pf_pool[n] for n in ss.pf_ms if n in ss.pf_pool]
    rows.sort(key=lambda r: r["combined"], reverse=True)

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
        for r in rows:
            c = band_color(r["band"])
            dig = r["digital"]
            digtxt = "n/a" if dig is None else f"{dig:.0f}"
            st.markdown(
                f'<div class="fa-row" style="display:grid;grid-template-columns:{grid};'
                f'gap:8px;padding:12px;margin-bottom:6px;border-radius:8px;align-items:center;'
                f'background:{c}14;border-left:4px solid {c}">'
                f'<div style="font-weight:600">{r["company"]}</div>'
                f'<div style="color:{theme.TEXT_DIM};font-size:0.85rem">{r["sector"]}</div>'
                f'<div style="font-weight:800;font-size:1.15rem;color:{c}">{r["combined"]:.0f}</div>'
                f'<div style="color:{theme.TEXT}">{r["financial"]:.0f}</div>'
                f'<div style="color:{theme.TEXT}">{digtxt}</div>'
                f'<div>{band_pill(r["band"])}</div></div>', unsafe_allow_html=True)


# ==================================================================== Live Lookup
# Non-financial listed companies only. Banks and NBFCs are deliberately excluded: the
# Altman Z-Score is built for industrial/commercial balance sheets and has no meaningful
# reading for a lender (no working capital, deposits are not "borrowings" in the Altman
# sense), so scoring them would be misleading rather than merely unavailable.
NSE_TOP = {
    "Reliance Industries": "RELIANCE", "TCS": "TCS", "Infosys": "INFY",
    "Hindustan Unilever": "HINDUNILVR", "ITC": "ITC", "Bharti Airtel": "BHARTIARTL",
    "Larsen & Toubro": "LT", "Asian Paints": "ASIANPAINT", "Maruti Suzuki": "MARUTI",
    "HCL Technologies": "HCLTECH", "Sun Pharma": "SUNPHARMA", "Titan": "TITAN", "Wipro": "WIPRO",
    "UltraTech Cement": "ULTRACEMCO", "Nestle India": "NESTLEIND", "Tata Steel": "TATASTEEL",
    "NTPC": "NTPC", "Power Grid": "POWERGRID",
    "Adani Enterprises": "ADANIENT", "Adani Ports": "ADANIPORTS", "Coal India": "COALINDIA",
    "JSW Steel": "JSWSTEEL", "Hindalco": "HINDALCO", "Tech Mahindra": "TECHM", "Grasim": "GRASIM",
    "Dr Reddys Labs": "DRREDDY", "Cipla": "CIPLA", "Britannia": "BRITANNIA",
    "Eicher Motors": "EICHERMOT", "Divis Labs": "DIVISLAB", "Hero MotoCorp": "HEROMOTOCO",
    "Apollo Hospitals": "APOLLOHOSP", "Tata Consumer": "TATACONSUM", "ONGC": "ONGC",
    "Vedanta": "VEDL", "Dabur": "DABUR",
    "Vodafone Idea": "IDEA", "MTNL": "MTNL", "Reliance Communications": "RCOM",
    "Suzlon Energy": "SUZLON", "SpiceJet": "SPICEJET",
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
            from foresight import (original_z, risk_from_original_z, band_for,
                                   score_company, recommend, india_distress_probability)
            st.write("Fetching financials and market cap from Screener ...")
            fin, mcap = screener_live.fetch_financials(ticker)
            fin.market_cap = mcap                    # so the engine uses the 1968 Altman Z
            z5, zone5, comps = original_z(fin, mcap)
            fin_risk = risk_from_original_z(z5)
            finsc = score_company(fin)               # FinancialScore, same engine as tracked
            model_p = india_distress_probability(fin)
            st.write("Fetching recent news and scoring sentiment ...")
            ev = live_news.news_evidence(name)
            heads = [c["text"] for c in ev["cited"]] + [q["text"] for q in ev["quiet"]]
            st.write("Reading the coverage for tone and severity ...")
            llm_sent = svc.live_llm_sentiment(name, tuple(heads))
            if llm_sent is not None:                 # magnitude-aware LLM read
                news_risk, avg_sent, news_rationale = llm_sent["risk"], llm_sent["tone"], llm_sent["rationale"]
            else:                                    # lexicon fallback (no key / offline)
                news_risk, avg_sent, news_rationale = ev["risk"], ev["tone"], ""
            combined = (round(0.6 * fin_risk + 0.4 * news_risk, 1)
                        if news_risk == news_risk else fin_risk)
            band = band_for(combined)
            recs = recommend(fin, finsc, None)
            st.write("Writing the analyst summary ...")
            term_pairs = tuple((lbl, cn) for lbl, _c, _v, cn in comps)
            summary = svc.live_analyst_summary(name, z5, zone5, combined, band, term_pairs,
                                               news_risk, news_rationale, model_p)
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
            st.markdown(_model_prob_html(model_p), unsafe_allow_html=True)
            body = summary or (f"Altman Z {z5:.2f} ({zone5}). "
                    + (f"{len(heads)} recent headlines, average tone {avg_sent:+.2f}."
                       if heads else "No recent news found."))
            st.markdown(f'<div class="fa-narrative" style="margin-top:12px">{body}</div>',
                        unsafe_allow_html=True)

    with panel():
        section_title("Financial Health")
        cards = svc.ratio_cards_fin(fin)
        for col, card in zip(st.columns(len(cards)), cards):
            color = theme.GOOD if card.good else theme.BAD
            col.markdown(f'<div class="fa-card"><div class="lbl">{card.label}</div>'
                         f'<div class="val" style="color:{color}">{card.value}</div>'
                         f'<div class="ctx">{card.context}</div></div>', unsafe_allow_html=True)

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
        maxc = max((abs(cn) for *_, cn in comps if cn == cn), default=1.0) or 1.0
        for lbl, coef, val, contrib in comps:
            nan = contrib != contrib  # NaN when a ratio can't be built (e.g. a bank)
            col = theme.GOOD if (not nan and contrib > 0) else theme.BAD
            w = 0 if nan else int(abs(contrib) / maxc * 220)
            vtxt = "n/a" if val != val else f"{val:+.2f}"
            ctxt = "n/a" if nan else f"{contrib:+.2f}"
            st.markdown(f'<div class="fa-term"><div class="tl">{lbl}</div>'
                        f'<div class="bar" style="width:{w}px;background:{col}"></div>'
                        f'<div class="tv">{coef} &times; {vtxt} = {ctxt}</div></div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div style="color:{theme.TEXT_DIM};font-size:0.8rem;margin-top:8px">'
                    'The original 1968 Altman Z, using the live market value of equity - available '
                    'because these are listed companies.</div>', unsafe_allow_html=True)

    if any(recs.values()):
        with panel():
            section_title("Recommendations")
            for aud in recs:
                if not recs[aud]:
                    continue
                st.markdown(f'<div class="fa-narrative" style="font-weight:600;margin-top:6px">'
                            f'{aud.value}</div>', unsafe_allow_html=True)
                for r in recs[aud]:
                    st.markdown(f'<div class="fa-signal"><div class="name">{r.title}</div>'
                                f'<div class="datum">{r.action}</div></div>',
                                unsafe_allow_html=True)

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
# `altman`   : real risk per fiscal year (the engine's own score on Screener financials).
# `foresight`: the COMPREHENSIVE score (Altman fused with signals). It carries a point at
#   every financial release (reason explains the Altman move) PLUS a point wherever a signal
#   moved it - so it is clear the financials are included, and the line runs end to end. It
#   sits at or above Altman in distress (signals add risk) and can dip below in a recovery
#   (signals clear risk before the accounts do).
TRACK_RECORD = {
    "Unitech (Real Estate)": {
        "sector": "Real Estate", "event_date": "2020-01-20", "event": "Board superseded, 2020",
        "lead": "Altman Safe until 2021; the comprehensive score was high from 2015.",
        "altman": [(2015, 13), (2016, 20), (2017, 21), (2018, 29), (2019, 34), (2020, 49),
                   (2021, 63), (2022, 69), (2023, 85)],
        "foresight": [
            ("2015-03-31", 28, "FY15 results: Altman Safe (13). The signals already add risk from mounting homebuyer delivery delays."),
            ("2015-11-01", 44, "Signal: consumer forums order refunds the company cannot fund."),
            ("2016-03-31", 48, "FY16 results: Altman still Safe (20); the complaints backlog keeps the comprehensive score high."),
            ("2017-04-26", 66, "FY17 filed (Altman still Safe, 21) - then weeks later promoters Sanjay and Ajay Chandra are arrested and management collapses; the score jumps."),
            ("2017-12-01", 72, "Signal: the Supreme Court intervenes over diverted homebuyer funds."),
            ("2018-03-31", 74, "FY18 results: Altman edges to Grey (29) as losses surface."),
            ("2019-03-31", 77, "FY19 results: Altman Grey (34); the signals kept the score high throughout."),
            ("2020-01-20", 83, "Signal: the Supreme Court supersedes the board; government management installed."),
            ("2021-06-30", 86, "FY21 results: Altman finally Distress (63) - the financials catch up to the signals."),
            ("2023-03-31", 90, "FY23 results: Altman Distress (85); both now agree the company has failed."),
        ],
        "takeaway": "For six years Altman read the accounts as Safe or Grey while the promoters were "
                    "arrested and the Supreme Court stepped in. The amber band is the risk those signals "
                    "carried - risk a financial-only score simply cannot see. This is the case for a "
                    "comprehensive read.",
    },
    "Future Retail (Retail)": {
        "sector": "Retail", "event_date": "2022-07-20", "event": "Insolvency (NCLT), 2022",
        "lead": "Altman Safe in FY2019; the signals had it elevated a year before.",
        "altman": [(2019, 15), (2020, 45), (2021, 85)],
        "foresight": [
            ("2019-03-31", 34, "FY19 results: Altman Safe (15) on lease-heavy but serviceable books; the signals flag the debt load."),
            ("2019-08-22", 44, "Signal: Amazon invests in Future Coupons with contested control rights."),
            ("2020-03-31", 60, "FY20 results: Altman slips to Grey (45) as COVID shuts malls and collections collapse."),
            ("2020-10-25", 74, "Signal: Amazon wins an arbitration that freezes the Reliance deal - a liquidity trap."),
            ("2021-03-31", 86, "FY21 results: Altman now Distress (85), converging with the comprehensive score."),
            ("2021-12-24", 90, "Signal: lenders reject the Reliance scheme."),
            ("2022-07-20", 95, "Insolvency: defaults on Rs 3,494 crore; admitted to NCLT."),
        ],
        "takeaway": "Altman scored Future Retail Safe in FY2019 - the year the control fight and the debt "
                    "problem were already public. The amber band shows the comprehensive score carrying "
                    "that risk a full year before the accounts admitted it.",
    },
    "Reliance Communications (Telecom)": {
        "sector": "Telecom", "event_date": "2019-02-01", "event": "Insolvency (NCLT), 2019",
        "lead": "Signals lifted the score above the accounts through 2016-17.",
        "altman": [(2015, 34), (2016, 60), (2017, 75), (2018, 95), (2019, 96)],
        "foresight": [
            ("2015-03-31", 44, "FY15 results: Altman Grey (34); the signals add risk as the tariff war begins."),
            ("2015-06-30", 50, "Signal: debt rising, refinancing pressure builds."),
            ("2016-03-31", 66, "FY16 results: Altman deteriorates to (60) as cash flows crater."),
            ("2016-09-05", 72, "Signal: Jio's free launch collapses industry pricing."),
            ("2017-03-31", 80, "FY17 results: Altman (75) - the losses are now in the accounts."),
            ("2017-05-30", 84, "Signal: debt downgraded to default grade after missed repayments."),
            ("2017-09-15", 88, "Signal: Ericsson files an insolvency plea over unpaid dues."),
            ("2018-03-31", 95, "FY18 results: Altman (95) - deep distress."),
            ("2019-02-01", 97, "Insolvency: the board opts for resolution through NCLT."),
        ],
        "takeaway": "Here Altman and the signals largely agree - but the comprehensive score still sits "
                    "above the accounts through 2016-17, because the rating cut and the Ericsson plea are "
                    "risk the financials had not yet booked.",
    },
    "Jaiprakash Associates (Infrastructure)": {
        "sector": "Infrastructure", "event_date": "2024-06-03", "event": "Insolvency (NCLT), 2024",
        "lead": "Altman swung up and down for years; the comprehensive score stayed high.",
        "altman": [(2016, 39), (2017, 82), (2018, 50), (2019, 67), (2020, 37), (2021, 42),
                   (2022, 46), (2023, 44), (2024, 49)],
        "foresight": [
            ("2016-03-31", 52, "FY16 results: Altman Grey (39); the signals add risk from the debt overhang."),
            ("2017-03-31", 86, "FY17 results: Altman spikes to Distress (82) on a bad year; the score is already there."),
            ("2017-06-01", 85, "Signal: named among the RBI's stressed accounts flagged for resolution."),
            ("2018-03-31", 74, "FY18 results: Altman rebounds to (50) - but the rating pressure keeps the score high."),
            ("2018-09-01", 78, "Signal: repeated rating downgrades; debt-recast talks stall."),
            ("2020-03-31", 74, "FY20 results: Altman swings back down to (37) - the accounts are volatile."),
            ("2021-03-01", 80, "Signal: lenders classify the loans as non-performing."),
            ("2022-09-01", 84, "Signal: rating agencies at default grade; asset monetisation slows."),
            ("2024-06-03", 90, "Insolvency: admitted to NCLT on a lender petition."),
        ],
        "takeaway": "Altman bounced between grey and distress year after year - impossible to act on. The "
                    "comprehensive score held steadily high because the rating and lender signals kept "
                    "pointing one way. The value here is a clear read through the noise.",
    },
    "Suzlon Energy (Renewables)": {
        "sector": "Renewables", "event_date": "2019-07-01", "event": "Near-default / recast, 2019",
        "lead": "In the recovery the signals cleared risk before the accounts did.",
        "altman": [(2017, 98), (2018, 96), (2019, 100), (2020, 100), (2021, 82), (2022, 91),
                   (2023, 6), (2024, 9), (2025, 12), (2026, 7)],
        "foresight": [
            ("2017-03-31", 98, "FY17 results: Altman deep distress (98); the signals concur on the debt load."),
            ("2018-03-31", 97, "FY18 results: Altman (96); ratings cut further."),
            ("2019-07-01", 99, "Signal: misses payments; a formal debt restructuring is invoked."),
            ("2020-03-31", 98, "FY20 results: Altman (100); COVID compounds the strain."),
            ("2021-03-31", 80, "FY21 results: Altman improves to (82) as debt falls."),
            ("2021-06-01", 74, "Signal: a large rights issue and deleveraging begin - the score turns down early."),
            ("2022-03-31", 72, "FY22 results: Altman still high (91) on lingering losses - the signals read the order-book recovery the accounts miss."),
            ("2022-11-01", 50, "Signal: return to operating profit; net debt falls sharply."),
            ("2023-03-31", 20, "FY23 results: Altman jumps to Safe (6); both now agree on the recovery."),
            ("2024-09-01", 12, "Signal: record order inflows; sustained profit."),
        ],
        "takeaway": "The score moves both ways. Through 2021-22 the green band shows the comprehensive "
                    "score falling below Altman - the rights issue and order-book recovery were visible in "
                    "the signals a year before the accounts confirmed the turnaround.",
    },
}

# Real annual closing share price (approx, in rupees), indexed to 100 at the window start.
# Annual points keep the line clean (no daily noise); delisted names have no clean intraday
# feed, so annual closes from public record are the honest, low-noise representation.
PRICE = {
    "Unitech (Real Estate)": [(2015, 100), (2016, 24), (2017, 28), (2018, 18), (2019, 10),
                              (2020, 5), (2021, 6), (2022, 4), (2023, 4)],
    "Future Retail (Retail)": [(2019, 100), (2020, 31), (2021, 13), (2022, 6)],
    "Reliance Communications (Telecom)": [(2015, 100), (2016, 69), (2017, 31), (2018, 23), (2019, 2)],
    "Jaiprakash Associates (Infrastructure)": [(2016, 100), (2017, 163), (2018, 225), (2019, 63),
                                               (2020, 18), (2021, 113), (2022, 88), (2023, 113), (2024, 225)],
    "Suzlon Energy (Renewables)": [(2017, 100), (2018, 58), (2019, 26), (2020, 11), (2021, 32),
                                   (2022, 47), (2023, 47), (2024, 221), (2025, 326), (2026, 289)],
}


def render_track_record() -> None:
    import datetime as _dt
    import bisect

    with panel():
        section_title("Track Record - the comprehensive score on real history")
        st.markdown('<div class="fa-headline">Altman is the financial-only benchmark credit desks have '
                    'used for decades - it is the blue base below. ForesightAI keeps it and adds the risk '
                    'the accounts miss: the amber band is what only the market signals see. Together they '
                    'are the comprehensive score. Hover any point for the reason it moved.</div>',
                    unsafe_allow_html=True)
        pick = st.selectbox("case", list(TRACK_RECORD), label_visibility="collapsed", key="tr_pick")

    d = TRACK_RECORD[pick]
    ALT = "#38BDF8"      # bright sky-blue for the Altman financial base
    axd = [_dt.date(y, 3, 31) for y, _ in d["altman"]]
    ay = [r for _, r in d["altman"]]
    axo = [x.toordinal() for x in axd]

    def _a_at(dt):
        t = dt.toordinal()
        if t <= axo[0]:
            return float(ay[0])
        if t >= axo[-1]:
            return float(ay[-1])
        i = bisect.bisect_right(axo, t) - 1
        f = (t - axo[i]) / (axo[i + 1] - axo[i])
        return ay[i] + f * (ay[i + 1] - ay[i])

    fx = [_dt.date.fromisoformat(dd) for dd, _, _ in d["foresight"]]
    fy = [float(s) for _, s, _ in d["foresight"]]
    reasons = [r for _, _, r in d["foresight"]]
    fxo = [x.toordinal() for x in fx]

    def _f_at(dt):
        t = dt.toordinal()
        if t <= fxo[0]:
            return fy[0]
        if t >= fxo[-1]:
            return fy[-1]
        i = bisect.bisect_right(fxo, t) - 1
        f = (t - fxo[i]) / (fxo[i + 1] - fxo[i])
        return fy[i] + f * (fy[i + 1] - fy[i])

    # Master grid = every filing date and every signal date, so both lines run end to end and
    # every marker sits on its line - no floating points.
    master = sorted(set(axd) | set(fx))
    m_alt = [_a_at(x) for x in master]
    m_for = [_f_at(x) for x in master]
    m_top = [max(a, s) for a, s in zip(m_alt, m_for)]
    m_low = [min(a, s) for a, s in zip(m_alt, m_for)]
    has_clear = any(s < a - 0.5 for a, s in zip(m_alt, m_for))

    fig = go.Figure()
    # Risk zone bands - the interpretation aid: low / medium / high.
    for lo, hi, col in [(0, 30, theme.GOOD), (30, 70, theme.WATCH), (70, 100, theme.BAD)]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.10, line_width=0, layer="below")
    for yy, lab, col in [(15, "LOW RISK", theme.GOOD), (50, "MEDIUM", theme.WATCH), (86, "HIGH RISK", theme.BAD)]:
        fig.add_annotation(xref="paper", x=0.006, y=yy, text=lab, showarrow=False, xanchor="left",
                           font=dict(color=col, size=9.5), opacity=0.9)

    # Blue shaded base = financial-only Altman risk (edge hidden; the clean line is drawn on top).
    fig.add_trace(go.Scatter(x=master, y=m_alt, mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(56,189,248,0.30)", showlegend=False, hoverinfo="skip"))
    # Amber band = the extra risk only the market signals see (the hero band).
    fig.add_trace(go.Scatter(x=master, y=m_top, name="Added risk: market signals", mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(251,146,60,0.58)", hoverinfo="skip"))
    if has_clear:
        fig.add_trace(go.Scatter(x=master, y=m_alt, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=master, y=m_low, name="Risk cleared by signals", mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(34,197,94,0.48)", hoverinfo="skip"))
    # Altman clean line + markers at every real filing - always connected.
    fig.add_trace(go.Scatter(x=axd, y=ay, name="Altman (financial-only)", mode="lines+markers",
                             line=dict(color=ALT, width=2.8),
                             marker=dict(size=7, color=ALT, line=dict(color="#0A1628", width=1)),
                             hovertemplate="FY%{x|%Y}: Altman risk %{y}<extra></extra>"))
    # ForesightAI comprehensive line + markers, a reason on every point.
    fig.add_trace(go.Scatter(x=fx, y=fy, name="ForesightAI (comprehensive)", mode="lines+markers",
                             line=dict(color="#FBBF24", width=4),
                             marker=dict(size=9, color="#FCD34D", line=dict(color="#FFFFFF", width=1.6)),
                             customdata=[[r] for r in reasons],
                             hovertemplate="<b>%{x|%d %b %Y} &middot; risk %{y:.0f}</b><br>%{customdata[0]}<extra></extra>"))
    px = PRICE.get(pick)
    if px:
        fig.add_trace(go.Scatter(x=[_dt.date(y, 3, 31) for y, _ in px], y=[v for _, v in px],
                                 name="Share price (annual close, indexed)", mode="lines", yaxis="y2",
                                 line=dict(color="#2DD4BF", width=2, dash="dot"),
                                 hovertemplate="FY%{x|%Y}: price index %{y}<extra></extra>"))
    ev = _dt.date.fromisoformat(d["event_date"])
    fig.add_vline(x=ev, line=dict(color="#F87171", width=1.6, dash="dash"))
    fig.add_annotation(x=ev, y=100, text=d["event"], showarrow=False, yanchor="top", xanchor="right",
                       font=dict(color="#F87171", size=11))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=34, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0C1D33",
                      font={"color": theme.TEXT},
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                      hoverlabel=dict(bgcolor=theme.BG_PANEL_2, font_size=12, align="left"),
                      yaxis=dict(range=[0, 100], title="Risk score (0-100)", gridcolor="rgba(255,255,255,0.06)"),
                      yaxis2=dict(title="Share price (annual close, indexed)", overlaying="y", side="right",
                                  showgrid=False, rangemode="tozero", color="#2DD4BF"),
                      xaxis=dict(title="", gridcolor="rgba(255,255,255,0.06)", dtick="M12", tickformat="%Y"))

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
                    'Blue is the financial-only Altman risk on each year of reported financials '
                    '(Screener.in); the amber band on top is the extra risk the market signals reveal, '
                    'and the top edge is the ForesightAI comprehensive score. Green marks where the '
                    'signals cleared risk before the accounts did. Share price is the annual closing '
                    'price, indexed to 100 at the start (annual points keep the line clean).</div>',
                    unsafe_allow_html=True)


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

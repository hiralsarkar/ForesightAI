"""Foresight AI visual system -- Bloomberg Terminal meets McKinsey risk memo.

The blueprint is explicit that default Streamlit styling must not ship. This module owns
the palette and injects the CSS that makes the app look like professional financial
software rather than a data-science demo.
"""

from __future__ import annotations

import streamlit as st

# Palette (AGENTS.md §2 -- do not deviate).
BG_BASE = "#0A1628"      # dark navy
BG_PANEL = "#0F1E33"     # slightly lifted panel
BG_PANEL_2 = "#152740"
ACCENT = "#F59E0B"       # amber
TEXT = "#FFFFFF"
TEXT_DIM = "#94A3B8"
BORDER = "#1E3350"

GOOD = "#22C55E"
WATCH = "#F59E0B"
ELEVATED = "#F97316"
BAD = "#EF4444"

BAND_COLOR = {
    "Healthy": GOOD,
    "Watch": WATCH,
    "Elevated Risk": ELEVATED,
    "Critical": BAD,
    "Unknown": TEXT_DIM,
}


def band_color(band: str) -> str:
    return BAND_COLOR.get(band, TEXT_DIM)


def inject() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .stApp {{ background: {BG_BASE}; color: {TEXT}; }}
        html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }}

        /* Kill Streamlit chrome */
        #MainMenu, header, footer {{ visibility: hidden; }}
        .block-container {{ padding-top: 2rem; max-width: 1200px; }}

        /* Brand */
        .fa-brand {{ font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; }}
        .fa-brand .amber {{ color: {ACCENT}; }}
        .fa-tagline {{ color: {TEXT_DIM}; font-size: 0.95rem; margin-top: -4px; }}

        /* Panels -- applied to Streamlit's native bordered container (st.container with a
           `fapanel-*` key emits a `.st-key-fapanel-*` class) so the styling actually wraps
           content. A bare <div class="fa-panel"> gets sanitized into an empty box and never
           contains the widgets that follow it, which is why the wrapper is used instead. */
        [class*="st-key-fapanel-"] {{
            background: {BG_PANEL}; border: 1px solid {BORDER} !important; border-radius: 14px !important;
            padding: 20px 22px !important; margin-bottom: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.25), 0 8px 24px -12px rgba(0,0,0,0.5);
            transition: border-color 160ms ease, box-shadow 160ms ease;
        }}
        [class*="st-key-fapanel-"]:hover {{
            border-color: {ACCENT}55 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.25), 0 12px 30px -12px rgba(0,0,0,0.6);
        }}
        .fa-panel {{
            background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
            padding: 20px 22px; margin-bottom: 16px;
        }}
        .fa-section-title {{
            font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: {TEXT_DIM}; margin-bottom: 14px;
        }}
        .fa-headline {{ font-size: 1.05rem; font-weight: 600; color: {TEXT}; margin-bottom: 4px; }}

        /* Metric cards */
        .fa-card {{
            background: {BG_PANEL_2}; border: 1px solid {BORDER}; border-radius: 10px;
            padding: 14px 16px; height: 100%;
            transition: transform 140ms ease, border-color 140ms ease;
        }}
        .fa-card:hover {{ transform: translateY(-2px); border-color: {ACCENT}55; }}
        .fa-signal {{ transition: background 140ms ease; }}
        .fa-signal:hover {{ background: {BG_PANEL_2}; }}
        .fa-row {{ transition: transform 140ms ease, filter 140ms ease; }}
        .fa-row:hover {{ transform: translateX(3px); filter: brightness(1.15); }}

        /* Tab bar -- amber active underline instead of Streamlit red */
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
        .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {TEXT_DIM}; }}
        .stTabs [aria-selected="true"] {{ color: {TEXT}; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background: {ACCENT}; }}
        .fa-card .lbl {{ color: {TEXT_DIM}; font-size: 0.72rem; text-transform: uppercase;
            letter-spacing: 0.05em; }}
        .fa-card .val {{ font-size: 1.5rem; font-weight: 700; margin: 4px 0; }}
        .fa-card .ctx {{ color: {TEXT_DIM}; font-size: 0.75rem; line-height: 1.35; }}

        /* Signal gauge rows */
        .fa-signal {{ display: flex; align-items: center; gap: 14px; padding: 12px 0;
            border-bottom: 1px solid {BORDER}; }}
        .fa-signal:last-child {{ border-bottom: none; }}
        .fa-dot {{ width: 12px; height: 12px; border-radius: 50%; flex: 0 0 12px; }}
        .fa-signal .name {{ font-weight: 600; width: 190px; flex: 0 0 190px; }}
        .fa-signal .datum {{ color: {TEXT_DIM}; font-size: 0.82rem; }}
        .fa-signal .pill {{ font-size: 0.7rem; font-weight: 700; padding: 2px 8px;
            border-radius: 20px; margin-left: auto; flex: 0 0 auto; }}

        .fa-band-pill {{ display: inline-block; padding: 3px 12px; border-radius: 20px;
            font-weight: 700; font-size: 0.8rem; }}

        /* Narrative box */
        .fa-narrative {{ background: {BG_PANEL_2}; border-left: 3px solid {ACCENT};
            padding: 14px 18px; border-radius: 6px; color: #E2E8F0; line-height: 1.5;
            font-size: 0.92rem; }}

        /* Waterfall term rows */
        .fa-term {{ display:flex; align-items:center; gap:10px; padding:6px 0; font-size:0.85rem; }}
        .fa-term .tl {{ width: 230px; flex: 0 0 230px; color:{TEXT}; }}
        .fa-term .bar {{ height: 16px; border-radius: 3px; }}
        .fa-term .tv {{ color:{TEXT_DIM}; font-size:0.8rem; }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{ background: {BG_PANEL}; border-right: 1px solid {BORDER}; }}

        /* Sliders in amber */
        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {ACCENT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def band_pill(band: str) -> str:
    c = band_color(band)
    return f'<span class="fa-band-pill" style="background:{c}22;color:{c};border:1px solid {c}66">{band}</span>'

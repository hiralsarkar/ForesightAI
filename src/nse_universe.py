"""The NSE universe the app can score on demand: display name -> Screener ticker.

Single source of truth shared by the Streamlit app (`app/main.py`) and the snapshot
builder (`research/build_snapshot.py`), so the baked `data/financials_snapshot.json`
never drifts from the list the UI offers.

Non-financial listed companies only. Banks and NBFCs are deliberately excluded: the
Altman Z-Score is built for industrial/commercial balance sheets and has no meaningful
reading for a lender (no working capital, deposits are not "borrowings" in the Altman
sense), so scoring them would be misleading rather than merely unavailable.
"""
from __future__ import annotations

NSE_TOP: dict[str, str] = {
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

"""Live news sentiment for any company: pull recent headlines from Google News and score
them with a distress-keyword lexicon tuned for stock-market headlines. No API key, so it
runs on the cloud. A production version would use a licensed news + sentiment feed.
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse

import requests

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Words that signal trouble vs strength in a market/results headline.
_NEG = set((
    "loss losses plunge plunges slump slumps fall falls fell drop drops decline declines weak "
    "weakness low lows lower lowered downgrade downgraded default defaults defaulted insolvency "
    "bankruptcy bankrupt nclt ibc fraud probe investigation scam resign resigns resignation "
    "layoff layoffs debt distress distressed crisis crash tumble tumbles sink sinks warning "
    "concern concerns questions lawsuit penalty fine delay miss missed stress stressed "
    "liquidation arrears overdue selloff worst struggle struggles trouble troubled halt "
    "suspended delist delisted cut cuts red negative dip dips slide slides").split())
_POS = set((
    "profit profits surge surges rise rises rose gain gains jump jumps rally rallies record high "
    "highs higher growth grow strong strength beat beats upgrade upgraded expand expansion order "
    "orders wins win won deal dividend bonus recovery recovers turnaround buyback outperform best "
    "robust soar soars boom raised raise").split())


def _score_headline(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def fetch_headlines(query: str, n: int = 15) -> list[str]:
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    xml = requests.get(url, headers=_UA, timeout=20).text
    heads, seen = [], set()
    for t in re.findall(r"<title>(.*?)</title>", xml, re.S):
        h = re.sub(r"\s+", " ", _html.unescape(t)).strip()
        if not h or h.lower() in {"google news", ""} or h.lower() in seen:
            continue
        seen.add(h.lower())
        heads.append(h)
        if len(heads) >= n:
            break
    return heads


def news_signal(company_name: str, n: int = 15):
    """Return (risk_0_100, avg_sentiment, headlines). Higher risk = more negative news."""
    heads = fetch_headlines(f"{company_name} share NSE", n)
    if not heads:
        return float("nan"), float("nan"), []
    avg = sum(_score_headline(h) for h in heads) / len(heads)   # [-1, 1], + = positive tone
    risk = round(max(0.0, -avg) * 100, 1)                       # negative news pushes risk up
    return risk, avg, heads

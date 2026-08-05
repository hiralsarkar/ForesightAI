"""Live news sentiment for any company: pull recent headlines from Google News and score
them for distress. No API key, so it runs on the cloud. A production version would use a
licensed news + sentiment feed.

Robustness: generic price-ticker headlines are dropped, distress words are weighted, and
the risk is the *share of directional headlines that are negative* (an average dilutes a
few bad headlines among many neutral ones), floored when a strong distress word appears.
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse

import requests

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Headlines that are just a price/ticker page carry no sentiment - drop them.
_NOISE = re.compile(r"share price|stock price|price today|price live|live nse|- google news", re.I)

_STRONG_NEG = set((
    "default defaults defaulted insolvency bankruptcy bankrupt nclt ibc fraud probe scam "
    "downgrade downgraded resign resigns resignation layoff layoffs plunge plunges crash "
    "tumble tumbles slump slumps sink sinks halt delist delisted liquidation distress crisis "
    "lawsuit penalty arrears write-off writeoff wilful defaulter").split())
_WEAK_NEG = set((
    "down fall falls fell drop drops decline declines loss losses weak weakness cut cuts "
    "concern concerns questions slide slides dip dips miss missed warning red lower lowered "
    "worst struggle struggles trouble troubled stress stressed selloff").split())
_STRONG_POS = set("surge surges soar soars rally rallies record upgrade upgraded wins won jump jumps boom".split())
_WEAK_POS = set((
    "profit profits rise rises rose gain gains growth strong deal dividend expand beat beats "
    "recovery recovers outperform robust raised order orders turnaround").split())


def _sentiment(text: str) -> tuple[int, bool]:
    """(sign, has_strong_distress_word): sign is -1 negative, +1 positive, 0 neutral."""
    tl = text.lower()
    words = re.findall(r"[a-z]+", tl)
    strong = any(w in _STRONG_NEG for w in words)
    neg = 2 * sum(w in _STRONG_NEG for w in words) + sum(w in _WEAK_NEG for w in words)
    pos = 2 * sum(w in _STRONG_POS for w in words) + sum(w in _WEAK_POS for w in words)
    if re.search(r"(down|slump|plunge|fall|drop|slid|sink|tank|crash)\s*\d", tl):
        neg += 2
    if re.search(r"(up|surge|jump|soar|rise|gain|rally|zoom)\s*\d", tl):
        pos += 2
    return (-1 if neg > pos else 1 if pos > neg else 0), strong


def fetch_headlines(query: str, n: int = 18) -> list[str]:
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    xml = requests.get(url, headers=_UA, timeout=20).text
    heads, seen = [], set()
    for t in re.findall(r"<title>(.*?)</title>", xml, re.S):
        h = re.sub(r"\s+", " ", _html.unescape(t)).strip()
        if not h or h.lower() == "google news" or _NOISE.search(h) or h.lower() in seen:
            continue
        seen.add(h.lower())
        heads.append(h)
        if len(heads) >= n:
            break
    return heads


def news_signal(company_name: str, n: int = 18):
    """Return (risk_0_100, net_tone, headlines). Higher risk = more negative news."""
    heads = fetch_headlines(f"{company_name} share NSE", n)
    if not heads:
        return float("nan"), float("nan"), []
    signs = [_sentiment(h) for h in heads]
    neg = sum(1 for s, _ in signs if s < 0)
    pos = sum(1 for s, _ in signs if s > 0)
    directional = neg + pos
    risk = round(100 * neg / directional, 1) if directional else 0.0
    if any(strong for _, strong in signs):        # a genuine distress word floors the signal
        risk = max(risk, 55.0)
    tone = round((pos - neg) / directional, 2) if directional else 0.0
    return risk, tone, heads

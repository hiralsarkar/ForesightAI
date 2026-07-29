"""News Sentiment signal (Module 2).

Built fallback-first, per the Phase 1 "never let the demo break" discipline. A
`SentimentScorer` interface has two implementations:

* `LoughranMcDonaldScorer` -- a no-dependency lexicon scorer using the finance-standard
  Loughran-McDonald word lists. General-purpose sentiment (VADER, etc.) mis-reads
  financial text -- "liability", "aggressive", "debt" are neutral-to-technical in
  finance -- so even the fallback uses the right vocabulary.
* `FinBertScorer` -- ProsusAI/finbert, loaded lazily. The blueprint's explicit choice;
  the volume here (~20 headlines x 8 companies) makes model latency a non-issue. Swapped
  in behind the same interface, so nothing downstream changes.

Headlines are real, dated coverage of the roster companies. The reading combines the
30-day mean polarity with its move against the prior 30 days, but the *level* dominates
the label when coverage is clearly positive or negative -- otherwise a company enjoying
successive ratings upgrades reads as "Deteriorating" on a small dip between two good
months.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from .base import SignalKind, SignalReading, clamp_score

# --- Loughran-McDonald lexicon (curated high-frequency subset) -------------------
# The full master dictionary has ~2,300 negative / ~350 positive terms; this is the
# high-signal subset relevant to distress coverage. Terms are stemmed loosely by
# substring match so "restructuring"/"restructure" both hit.
_LM_NEGATIVE = frozenset({
    "loss", "losses", "decline", "declining", "declined", "default", "defaulted",
    "bankrupt", "bankruptcy", "insolvency", "insolvent", "restructur", "distress",
    "downgrade", "downgraded", "litigation", "lawsuit", "probe", "investigation",
    "fraud", "misstatement", "resign", "resigned", "resignation", "exit", "layoff",
    "layoffs", "shortfall", "deficit", "delinquent", "arrears", "impairment",
    "writedown", "writeoff", "liquidation", "moratorium", "downturn", "slump",
    "plunge", "plummet", "crisis", "distressed", "unpaid", "overdue", "breach",
    "covenant", "downsizing", "grounded", "halt", "suspend", "suspended", "delay",
    "delayed", "cut", "cuts", "weak", "weakness", "concern", "concerns", "warning",
    "risk", "risky", "burden", "strain", "struggle", "struggling", "distraught",
    "nonpayment", "unable", "failure", "failed", "negative", "deteriorat",
})
_LM_POSITIVE = frozenset({
    "profit", "profitable", "growth", "grew", "gain", "gains", "strong", "strength",
    "record", "surge", "surged", "upgrade", "upgraded", "beat", "outperform",
    "expansion", "expanding", "robust", "improve", "improved", "improvement",
    "rebound", "recovery", "recovered", "healthy", "resilient", "milestone",
    "award", "leading", "leader", "efficient", "dividend", "bonus", "success",
    "successful", "win", "winning", "optimistic", "upbeat", "solid",
})


@runtime_checkable
class SentimentScorer(Protocol):
    """Maps one headline to a polarity in [-1, +1] (+1 positive, -1 negative)."""

    def score(self, text: str) -> float: ...

    @property
    def name(self) -> str: ...


class LoughranMcDonaldScorer:
    """No-dependency finance-lexicon scorer. The default, always available."""

    name = "Loughran-McDonald"

    def score(self, text: str) -> float:
        t = text.lower()
        pos = sum(1 for w in _LM_POSITIVE if w in t)
        neg = sum(1 for w in _LM_NEGATIVE if w in t)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)


class FinBertScorer:
    """ProsusAI/finbert, loaded lazily. Requires torch + transformers.

    Kept behind the same interface as the fallback so swapping it in changes nothing
    downstream. Construction does not import torch; the first `score()` does.
    """

    name = "FinBERT"

    def __init__(self, model_id: str = "ProsusAI/finbert") -> None:
        self.model_id = model_id
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline  # heavy import, deferred

            self._pipe = pipeline("sentiment-analysis", model=self.model_id, top_k=None)
        return self._pipe

    def score(self, text: str) -> float:
        pipe = self._ensure()
        scores = {d["label"].lower(): d["score"] for d in pipe(text)[0]}
        # FinBERT emits positive / negative / neutral; polarity is the signed difference.
        return float(scores.get("positive", 0.0) - scores.get("negative", 0.0))


class CachedScorer:
    """Replays FinBERT's polarity from a JSON cache: {headline text: score}.

    Lets the deployed app reproduce FinBERT's exact numbers with no torch/transformers
    installed, the same pre-cache-for-offline pattern the narratives use. A cache miss
    (a headline not seen at build time) falls back to the lexicon, so a new headline
    never crashes the app.
    """

    name = "FinBERT (cached)"

    def __init__(self, table: dict, fallback: "SentimentScorer | None" = None) -> None:
        self._table = table
        self._fallback = fallback or LoughranMcDonaldScorer()

    def score(self, text: str) -> float:
        v = self._table.get(text)
        return float(v) if v is not None else self._fallback.score(text)


_SENTIMENT_CACHE = Path(__file__).resolve().parents[2] / "data" / "demo" / "sentiment_cache.json"


def load_sentiment_cache() -> dict | None:
    """The precomputed FinBERT scores, or None if absent."""
    try:
        return json.loads(_SENTIMENT_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None


def prewarm_sentiment_cache() -> Path:
    """Score every demo headline with FinBERT and persist the table (needs torch)."""
    from . import demo_signals as d

    fb = FinBertScorer()
    table: dict[str, float] = {}
    for company in d._DATA.values():
        for h in company.get("headlines", []):
            table[h.text] = round(fb.score(h.text), 6)
    _SENTIMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _SENTIMENT_CACHE.write_text(json.dumps(table, indent=1), encoding="utf-8")
    return _SENTIMENT_CACHE


def default_scorer(prefer_finbert: bool = True) -> SentimentScorer:
    """Best available scorer, resolved cache -> FinBERT -> lexicon.

    Validated (docs/phase2_findings.md): FinBERT and the lexicon agree on sign, but the
    lexicon reads several distress headlines *neutral* ("defers loan repayment") because
    they contain no lexicon words, while FinBERT reads them negative from context. So the
    deployed app serves FinBERT's exact scores from the cache (no torch needed); locally,
    live FinBERT is used if the cache is absent; the lexicon is the never-breaks fallback.
    """
    if prefer_finbert:
        cache = load_sentiment_cache()
        if cache:
            return CachedScorer(cache)
        try:
            import importlib.util

            if importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"):
                return FinBertScorer()
        except Exception:
            pass
    return LoughranMcDonaldScorer()


@dataclass(frozen=True)
class Headline:
    published: date
    text: str
    source: str = ""


def _polarity_to_risk(polarity: float) -> float:
    """Map mean polarity [-1,+1] to a 0-100 risk score.

    Anchored so healthy-ish coverage (polarity ~ +0.5) lands Healthy (~18), neutral
    lands mid-Watch (~40), and uniformly negative coverage lands Critical (~85).
    """
    return clamp_score(40.0 - 45.0 * polarity)


def _mean_polarity(headlines: list[Headline], scorer: SentimentScorer,
                   start: date, end: date) -> float | None:
    window = [h for h in headlines if start < h.published <= end]
    if not window:
        return None
    return sum(scorer.score(h.text) for h in window) / len(window)


def score_as_of(
    headlines: list[Headline],
    when: date,
    scorer: SentimentScorer | None = None,
    company: str = "",
    window_days: int = 30,
) -> SignalReading:
    """News-sentiment reading as of `when`: 30-day mean polarity + trend vs prior 30.

    Point-in-time: only headlines published on or before `when` are considered.
    """
    scorer = scorer or LoughranMcDonaldScorer()
    prior_start = when - timedelta(days=2 * window_days)
    mid = when - timedelta(days=window_days)

    recent = _mean_polarity(headlines, scorer, mid, when)
    previous = _mean_polarity(headlines, scorer, prior_start, mid)

    if recent is None:
        return SignalReading(
            kind=SignalKind.NEWS_SENTIMENT, as_of=when, risk_score=40.0,
            label="No coverage", datum="No news coverage in the last 30 days.", raw=0.0,
        )

    risk = _polarity_to_risk(recent)

    # Direction, and the specific datum the gauge shows.
    n_recent = len([h for h in headlines if mid < h.published <= when])
    # The level dominates the trend when coverage is clearly positive or clearly negative.
    # Without this, a company with strongly positive coverage (e.g. successive ratings
    # upgrades) gets labelled "Deteriorating" on a small dip between two good months.
    if recent > 0.2:
        label, moved = "Positive", ("less emphatic than" if previous is not None
                                    and recent - previous < -0.15 else "in line with")
    elif recent < -0.2:
        label, moved = "Negative", ("sharply more negative than" if previous is not None
                                    and recent - previous < -0.15 else "in line with")
    elif previous is not None and recent - previous < -0.15:
        label, moved = "Deteriorating", "sharply more negative than"
    elif previous is not None and recent - previous > 0.15:
        label, moved = "Improving", "more positive than"
    else:
        label, moved = "Stable", "broadly in line with"

    tone = "negative" if recent < -0.1 else "positive" if recent > 0.1 else "mixed"
    datum = (
        f"{n_recent} headlines in the last 30 days, tone {tone} "
        f"(polarity {recent:+.2f}); coverage {moved} the prior month."
    )

    return SignalReading(
        kind=SignalKind.NEWS_SENTIMENT, as_of=when, risk_score=risk,
        label=label, datum=datum, raw=float(recent),
    )

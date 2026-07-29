"""Shared data model for the four Digital Pulse signals (Module 2).

Design decisions locked in here, per AGENTS.md and the Phase 2 architecture:

* **Same 0-100 risk scale and band language as the financial score.** Higher = worse,
  bands Healthy / Watch / Elevated Risk / Critical. This is what makes Module 3 fusion a
  plain weighted average later -- financial and digital scores speak one language.

* **Every reading carries a specific explanatory datum**, not a restated metric. The
  blueprint's line: not "Sentiment: negative" but "Sentiment declining for 11 weeks;
  68% of coverage mentions 'liquidity concerns'." The datum is a required field.

* **Time-series, not snapshot.** A signal is observed at multiple dates. The latest
  reading feeds the Module 2 gauges; the trajectory feeds the case-study timeline
  (Slide 4). `SignalSeries` holds the history; `.latest()` is what the gauge shows.

* **Trend is first-class**, computed from the series, because for these signals the
  direction matters more than the level (a company going 300 -> 80 job postings is the
  signal, not the absolute 80).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class SignalKind(str, Enum):
    NEWS_SENTIMENT = "News Sentiment"
    LEADERSHIP = "Leadership Stability"
    HIRING = "Hiring Activity"
    EMPLOYEE = "Employee Confidence"


class Trend(str, Enum):
    """Direction of travel. Worsening is what a risk desk cares about."""

    IMPROVING = "Improving"
    STABLE = "Stable"
    DETERIORATING = "Deteriorating"


# Reuse the single band definition from the serving financial score so the digital
# gauges and the financial gauge cannot drift apart.
def band_for(risk: float) -> str:
    from ..serving.financial_score import band_for as _bf

    return _bf(risk)


@dataclass(frozen=True)
class SignalReading:
    """One signal, observed at one date, on the shared risk scale."""

    kind: SignalKind
    as_of: date
    risk_score: float          # 0-100, higher = worse
    label: str                 # signal-specific status, e.g. "Contracting"
    datum: str                 # the specific explanatory sentence (required)
    raw: float = 0.0           # underlying measure (sentiment, exit count, ...)
    #: True for a verified hard event -- a tribunal displacing the board, an auditor
    #: resigning. These are *facts*, not inferences, and must not be averaged away by
    #: softer signals (see `composite.combine`). News tone in particular runs the wrong
    #: way on legal process: "NCLAT stays insolvency admission" reads +0.69 positive to
    #: FinBERT while describing a company in insolvency.
    hard_event: bool = False

    @property
    def band(self) -> str:
        return band_for(self.risk_score)


@dataclass
class SignalSeries:
    """A signal's full observed history for one company."""

    kind: SignalKind
    company: str
    readings: list[SignalReading] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.readings.sort(key=lambda r: r.as_of)

    def latest(self) -> Optional[SignalReading]:
        return self.readings[-1] if self.readings else None

    def as_of(self, when: date) -> Optional[SignalReading]:
        """Point-in-time accessor: the most recent reading at or before `when`.

        This is the Trap-3 guard -- scoring a company as of a past date must never see a
        reading published after it.
        """
        eligible = [r for r in self.readings if r.as_of <= when]
        return eligible[-1] if eligible else None

    def trend(self, window: int = 2) -> Trend:
        """Direction over the last `window` readings, on the risk scale."""
        if len(self.readings) < 2:
            return Trend.STABLE
        recent = self.readings[-window:]
        delta = recent[-1].risk_score - recent[0].risk_score
        if delta > 5:
            return Trend.DETERIORATING
        if delta < -5:
            return Trend.IMPROVING
        return Trend.STABLE


def clamp_score(x: float) -> float:
    return float(min(max(x, 0.0), 100.0))

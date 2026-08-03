"""Digital Pulse composite.

Fuses the four signal readings into one 0-100 digital risk score on the same scale and
band language as the financial score, so that fusing financial and digital is a
plain weighted average later.

**Scope boundary:** this module stops at the digital composite. It does NOT combine with
the financial score -- that fusion is a later step. Building it here would be drift.

Weights reflect signal reliability: leadership
(dated public filings, no selection-bias risk) and news sentiment carry the signal;
hiring and employee reviews are soft and illustrative, so they inform but do not drive.
Missing signals are dropped and the remaining weights renormalised, so a company with no
review data is not penalised for the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .base import SignalKind, SignalReading, band_for, clamp_score

_WEIGHTS: dict[SignalKind, float] = {
    SignalKind.LEADERSHIP: 0.35,
    SignalKind.NEWS_SENTIMENT: 0.35,
    SignalKind.HIRING: 0.15,
    SignalKind.EMPLOYEE: 0.15,
}


@dataclass
class DigitalPulse:
    """The four gauges plus their composite, as of one date."""

    company: str
    as_of: date
    readings: list[SignalReading] = field(default_factory=list)
    composite_score: float = 0.0

    @property
    def band(self) -> str:
        return band_for(self.composite_score)

    def by_kind(self, kind: SignalKind) -> Optional[SignalReading]:
        for r in self.readings:
            if r.kind is kind:
                return r
        return None

    def top_concern(self) -> Optional[SignalReading]:
        """The single most alarming signal -- what the narrative should lead with."""
        return max(self.readings, key=lambda r: r.risk_score) if self.readings else None


def combine(company: str, as_of: date, readings: list[SignalReading]) -> DigitalPulse:
    """Weighted-average the available signal readings into a digital composite."""
    present = [r for r in readings if r is not None]
    total_w = sum(_WEIGHTS.get(r.kind, 0.0) for r in present)

    if total_w == 0:
        score = 0.0
    else:
        score = clamp_score(
            sum(r.risk_score * _WEIGHTS.get(r.kind, 0.0) for r in present) / total_w
        )

    # Hard-event floor. A verified fact (a tribunal displacing the board, an auditor
    # walking out) cannot be diluted below its own reading by softer signals. Without
    # this, Reliance Comm's board-suspension reading of 96 was averaged against a news
    # score of 25 -- because FinBERT reads "NCLAT stays insolvency admission" as
    # *positive* -- and the company de-escalated from Critical to Elevated purely by
    # adding data. Facts floor inferences; that is standard credit practice for a
    # covenant breach or insolvency filing.
    hard = [r.risk_score for r in present if r.hard_event]
    if hard:
        score = max(score, max(hard))

    return DigitalPulse(
        company=company, as_of=as_of, readings=list(present), composite_score=round(score, 1)
    )

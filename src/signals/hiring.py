"""Hiring Activity signal (soft).

Active job-posting count over time (Naukri-style). The
*trend* matters more than the absolute -- a company going 300 -> 80 postings is signalling
contraction regardless of its size. Historical posting counts cannot be scraped after the
fact, so the demo data is illustrative and labelled as such (same basis as Glassdoor).

Point-in-time via `score_as_of`: only counts observed on or before `when` are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .base import SignalKind, SignalReading, clamp_score


@dataclass(frozen=True)
class HiringObservation:
    observed: date
    active_postings: int


def _pct_change(latest: int, baseline: int) -> float:
    if baseline == 0:
        return 0.0
    return (latest - baseline) / baseline


def score_as_of(
    observations: list[HiringObservation],
    when: date,
    company: str = "",
) -> SignalReading:
    """Hiring reading as of `when`, from the trend across prior observations."""
    hist = sorted([o for o in observations if o.observed <= when], key=lambda o: o.observed)
    if not hist:
        return SignalReading(
            kind=SignalKind.HIRING, as_of=when, risk_score=40.0,
            label="No data", datum="No hiring data available.", raw=0.0,
        )

    latest = hist[-1]
    baseline = hist[0]
    change = _pct_change(latest.active_postings, baseline.active_postings)

    # Contraction raises risk; expansion lowers it. Anchored so a ~60% contraction lands
    # Elevated and steady/growing hiring lands Healthy.
    risk = clamp_score(35.0 - 200.0 * change)

    if change <= -0.08:
        label = "Contracting"
    elif change >= 0.05:
        label = "Growing"
    else:
        label = "Stable"

    datum = (
        f"Headcount {baseline.active_postings:,} -> {latest.active_postings:,} "
        f"({change:+.0%}) since {baseline.observed.isoformat()}."
    )
    return SignalReading(
        kind=SignalKind.HIRING, as_of=when, risk_score=risk,
        label=label, datum=datum, raw=float(latest.active_postings),
    )

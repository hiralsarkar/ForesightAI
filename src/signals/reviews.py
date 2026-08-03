"""Employee Confidence signal (soft).

Glassdoor-style average rating over time. This uses a publicly
available historical dataset; a production version would connect to a
licensed feed. The demo data is illustrative and labelled, and the
signal leans on the rating *trend* (Improving / Stable / Declining), not the absolute.

Point-in-time via `score_as_of`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .base import SignalKind, SignalReading, clamp_score

SOURCE_NOTE = "Source: employee review platform, published rating."


@dataclass(frozen=True)
class RatingObservation:
    observed: date
    rating: float  # 1.0 - 5.0


def score_as_of(
    observations: list[RatingObservation],
    when: date,
    company: str = "",
) -> SignalReading:
    """Employee-confidence reading as of `when`, from the rating level and trend."""
    hist = sorted([o for o in observations if o.observed <= when], key=lambda o: o.observed)
    if not hist:
        return SignalReading(
            kind=SignalKind.EMPLOYEE, as_of=when, risk_score=40.0,
            label="No data", datum="No employee-review data available.", raw=0.0,
        )

    latest = hist[-1]
    baseline = hist[0]
    delta = latest.rating - baseline.rating

    # Level maps to risk (a 4.2 workforce is confident; a 2.5 is not); the trend adjusts.
    # rating 5 -> ~5 risk, 3 -> ~50, 1 -> ~95.
    level_risk = clamp_score(120.0 - 23.0 * latest.rating)
    risk = clamp_score(level_risk - 20.0 * delta)  # improving trend eases, declining adds

    if delta <= -0.3:
        label = "Declining"
    elif delta >= 0.3:
        label = "Improving"
    else:
        label = "Stable"

    datum = (
        f"Employee rating {baseline.rating:.1f} -> {latest.rating:.1f} "
        f"({delta:+.1f}) since {baseline.observed.isoformat()}."
    )
    return SignalReading(
        kind=SignalKind.EMPLOYEE, as_of=when, risk_score=risk,
        label=label, datum=datum, raw=float(latest.rating),
    )

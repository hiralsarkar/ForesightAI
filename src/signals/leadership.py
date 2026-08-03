"""Leadership Stability signal from BSE-style corporate filings (anchor).

This is the hardest-evidence of the four signals, and it is
uniquely India-specific: every board-level change, KMP resignation, and auditor change
must be filed with the exchange, dated and public. There is **no selection-bias risk** --
a resignation happened on its filing date or it did not. That makes this the anchor the
softer signals lean on.

The signal is the count of *senior* departures in a trailing window (default 6 months),
weighted by seniority: a CFO or auditor walking out is a sharper distress tell than an
independent director rotating off. More than two senior exits in the window is the red
line, and it maps to the Elevated/Critical threshold.

Point-in-time is intrinsic: `score_as_of(when)` only counts events filed on or before
`when`, so a 2019 assessment never sees a 2020 filing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from .base import SignalKind, SignalReading, clamp_score


class Role(str, Enum):
    CEO = "CEO"
    CFO = "CFO"
    MD = "Managing Director"
    CHAIRMAN = "Chairman"
    FOUNDER = "Founder"
    AUDITOR = "Auditor"
    COMPANY_SECRETARY = "Company Secretary"
    WHOLETIME_DIRECTOR = "Whole-Time Director"
    INDEPENDENT_DIRECTOR = "Independent Director"
    BOARD = "Board of Directors"


class EventType(str, Enum):
    RESIGNATION = "resignation"
    APPOINTMENT = "appointment"
    AUDITOR_EXIT = "auditor exit"
    #: Insolvency admission vesting the board's powers in a resolution professional.
    #: The most severe governance event there is -- the board stops governing.
    BOARD_SUSPENSION = "board powers suspended"


#: Seniority weights. A departure's contribution to the risk score. Auditor and CFO
#: exits are weighted highest -- in Indian corporate collapses (IL&FS, many NBFCs) an
#: auditor walking out preceded the public crisis.
_EXIT_WEIGHT: dict[Role, float] = {
    Role.BOARD: 5.0,   # whole board displaced by a resolution professional
    Role.AUDITOR: 2.0,
    Role.CFO: 1.8,
    Role.CEO: 1.6,
    Role.MD: 1.6,
    Role.FOUNDER: 1.5,
    Role.CHAIRMAN: 1.4,
    Role.WHOLETIME_DIRECTOR: 1.1,
    Role.COMPANY_SECRETARY: 0.9,
    Role.INDEPENDENT_DIRECTOR: 0.8,
}

# Points per unit of weighted-exit, tuned so "> 2 exits = red" lands in
# the Elevated band: two average senior exits (weight ~1.5 each = 3.0) -> ~54.
_POINTS_PER_WEIGHT = 18.0
_BASELINE = 6.0  # a stable company still shows minor churn


@dataclass(frozen=True)
class LeadershipEvent:
    company: str
    filing_date: date
    role: Role
    person: str
    event_type: EventType
    #: True only for events confirmed against a real, dated public filing. The anchor's
    #: whole value ("a resignation happened on its date or it did not") depends
    #: on this being real -- an unverified stand-in must not be presented as a filing.
    verified: bool = True

    @property
    def is_exit(self) -> bool:
        return self.event_type in (
            EventType.RESIGNATION, EventType.AUDITOR_EXIT, EventType.BOARD_SUSPENSION)


def _window_exits(
    events: list[LeadershipEvent], when: date, months: int
) -> list[LeadershipEvent]:
    start = when - timedelta(days=int(months * 30.44))
    return [
        e for e in events
        if e.is_exit and start <= e.filing_date <= when
    ]


def _describe(exits: list[LeadershipEvent], months: int) -> str:
    """The specific explanatory datum -- who left, not just how many."""
    if not exits:
        return f"No senior departures filed in the last {months} months."

    # A board suspension is not a "departure" -- describe it as what it is.
    suspensions = [e for e in exits if e.event_type is EventType.BOARD_SUSPENSION]
    if suspensions:
        others = len(exits) - len(suspensions)
        extra = f" Also {others} senior departure{'s' if others > 1 else ''} in the period." if others else ""
        d = suspensions[0].filing_date.isoformat()
        return (f"Board powers suspended and vested in a resolution professional "
                f"({d}) - the most severe governance event on record.{extra}")

    roles = [e.role.value for e in exits]
    # Compress duplicates: "2 Independent Directors" rather than listing each.
    counts: dict[str, int] = {}
    for r in roles:
        counts[r] = counts.get(r, 0) + 1
    parts = [f"{n} {r}{'s' if n > 1 else ''}" if n > 1 else r for r, n in counts.items()]
    joined = ", ".join(parts)
    return f"{len(exits)} senior departures in {months} months: {joined}."


def score_as_of(
    events: list[LeadershipEvent],
    when: date,
    company: str = "",
    months: int = 12,
) -> SignalReading:
    """Leadership-stability reading as of `when`, counting only prior filings."""
    exits = _window_exits(events, when, months)
    weighted = sum(_EXIT_WEIGHT.get(e.role, 1.0) for e in exits)
    risk = clamp_score(_BASELINE + weighted * _POINTS_PER_WEIGHT)

    n = len(exits)
    if any(e.event_type is EventType.BOARD_SUSPENSION for e in exits):
        label = "Board displaced"      # not "churn" -- the board stopped governing
    elif n == 0:
        label = "Stable"
    elif n <= 2:
        label = "Some churn"
    else:
        label = "High turnover"  # the "> 2 = red" case

    # A verified tribunal order or auditor exit is a hard fact, flagged so the composite
    # cannot average it away against softer, tone-based signals.
    hard = any(
        e.verified and e.event_type in (EventType.BOARD_SUSPENSION, EventType.AUDITOR_EXIT)
        for e in exits
    )
    return SignalReading(
        kind=SignalKind.LEADERSHIP,
        as_of=when,
        risk_score=risk,
        label=label,
        datum=_describe(exits, months),
        raw=float(n),
        hard_event=hard,
    )

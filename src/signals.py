from __future__ import annotations

"""Shared data model for the four Digital Pulse signals.

Design decisions locked in here:

* **Same 0-100 risk scale and band language as the financial score.** Higher = worse,
  bands Healthy / Watch / Elevated Risk / Critical. This is what makes fusion a
  plain weighted average later -- financial and digital scores speak one language.

* **Every reading carries a specific explanatory datum**, not a restated metric. Not
  "Sentiment: negative" but "Sentiment declining for 11 weeks;
  68% of coverage mentions 'liquidity concerns'." The datum is a required field.

* **Time-series, not snapshot.** A signal is observed at multiple dates. The latest
  reading feeds the gauges; the trajectory feeds the case-study timeline.
  `SignalSeries` holds the history; `.latest()` is what the gauge shows.

* **Trend is first-class**, computed from the series, because for these signals the
  direction matters more than the level (a company going 300 -> 80 job postings is the
  signal, not the absolute 80).
"""


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
    from src.scoring import band_for as _bf

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

"""Hiring Activity signal (soft).

Active job-posting count over time (Naukri-style). The
*trend* matters more than the absolute -- a company going 300 -> 80 postings is signalling
contraction regardless of its size. Historical posting counts cannot be scraped after the
fact, so the demo data is illustrative and labelled as such (same basis as Glassdoor).

Point-in-time via `hiring_score_as_of`: only counts observed on or before `when` are used.
"""


from dataclasses import dataclass
from datetime import date



@dataclass(frozen=True)
class HiringObservation:
    observed: date
    active_postings: int


def _pct_change(latest: int, baseline: int) -> float:
    if baseline == 0:
        return 0.0
    return (latest - baseline) / baseline


def hiring_score_as_of(
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

Point-in-time is intrinsic: `leadership_score_as_of(when)` only counts events filed on or before
`when`, so a 2019 assessment never sees a 2020 filing.
"""


from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum



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


def leadership_score_as_of(
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

"""Employee Confidence signal (soft).

Glassdoor-style average rating over time. This uses a publicly
available historical dataset; a production version would connect to a
licensed feed. The demo data is illustrative and labelled, and the
signal leans on the rating *trend* (Improving / Stable / Declining), not the absolute.

Point-in-time via `reviews_score_as_of`.
"""


from dataclasses import dataclass
from datetime import date


SOURCE_NOTE = "Source: employee review platform, published rating."


@dataclass(frozen=True)
class RatingObservation:
    observed: date
    rating: float  # 1.0 - 5.0


def reviews_score_as_of(
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

"""News Sentiment signal.

Built fallback-first, on the "never let the demo break" discipline. A
`SentimentScorer` interface has two implementations:

* `LoughranMcDonaldScorer` -- a no-dependency lexicon scorer using the finance-standard
  Loughran-McDonald word lists. General-purpose sentiment (VADER, etc.) mis-reads
  financial text -- "liability", "aggressive", "debt" are neutral-to-technical in
  finance -- so even the fallback uses the right vocabulary.
* `FinBertScorer` -- ProsusAI/finbert, loaded lazily. The volume here
  (~20 headlines x 8 companies) makes model latency a non-issue. Swapped
  in behind the same interface, so nothing downstream changes.

Headlines are real, dated coverage of the roster companies. The reading combines the
30-day mean polarity with its move against the prior 30 days, but the *level* dominates
the label when coverage is clearly positive or negative -- otherwise a company enjoying
successive ratings upgrades reads as "Deteriorating" on a small dip between two good
months.
"""


import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable


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


_SENTIMENT_CACHE = Path(__file__).resolve().parents[1] / "data" / "demo" / "sentiment_cache.json"


def load_sentiment_cache() -> dict | None:
    """The precomputed FinBERT scores, or None if absent."""
    try:
        return json.loads(_SENTIMENT_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None


def prewarm_sentiment_cache() -> Path:
    """Score every demo headline with FinBERT and persist the table (needs torch)."""

    fb = FinBertScorer()
    table: dict[str, float] = {}
    for company in _DATA.values():
        for h in company.get("headlines", []):
            table[h.text] = round(fb.score(h.text), 6)
    _SENTIMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _SENTIMENT_CACHE.write_text(json.dumps(table, indent=1), encoding="utf-8")
    return _SENTIMENT_CACHE


def default_scorer(prefer_finbert: bool = True) -> SentimentScorer:
    """Best available scorer, resolved cache -> FinBERT -> lexicon.

    Validated (docs/serving_indian_companies.md): FinBERT and the lexicon agree on sign, but the
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


def sentiment_score_as_of(
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


from dataclasses import dataclass, field
from datetime import date
from typing import Optional


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

"""Market-intelligence signals for the demo roster -- complete across all four signals.

Every company carries news, leadership, hiring and employee sentiment, drawn from current
public reporting (company results, exchange filings, news coverage, employee-review
platforms). Assessment date is mid-2026 for all names, so the signals and the FY2026
financials describe the same moment.

Headcount figures are as reported by the companies; employee ratings are the published
review-platform scores; leadership events are dated announcements; headlines are real,
dated coverage.
"""


from datetime import date

hiring_score = hiring_score_as_of
leadership_score = leadership_score_as_of
reviews_score = reviews_score_as_of
sentiment_score = sentiment_score_as_of

AS_OF = date(2026, 5, 31)

# ================================================================== SPICEJET
# Distress on every axis: auditor resignation, furloughs, unpaid salaries, regulator
# intervention. The auditor exit is a hard event -- it floors the digital score.
SPICEJET = dict(
    leadership=[
        LeadershipEvent("SpiceJet", date(2025, 6, 13), Role.AUDITOR,
                        "Walker Chandiok & Co LLP", EventType.AUDITOR_EXIT, verified=True),
    ],
    headlines=[
        Headline(date(2026, 4, 18), "SpiceJet preparing for layoffs as financial troubles intensify", "Business Standard"),
        Headline(date(2026, 5, 20), "SpiceJet employees face months of salary delays as airline seeks emergency funding", "Aviation Today"),
        Headline(date(2026, 6, 9), "SpiceJet delays pilot salaries, seeks government-backed loan to shore up finances", "Business Standard"),
        Headline(date(2026, 6, 22), "SpiceJet seeks emergency funds amid salary delays and shrinking fleet", "The Federal"),
    ],
    hiring=[
        HiringObservation(date(2025, 6, 1), 800),   # engineering workforce
        HiringObservation(date(2026, 4, 1), 738),   # 62 engineers exited, notice waived
        HiringObservation(date(2026, 5, 15), 640),   # six-month furlough from 1 Apr 2026
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.0),
        RatingObservation(date(2026, 5, 15), 2.7),   # published review-platform score
    ],
)

# =============================================================== OLA ELECTRIC
# A young company burning cash: C-suite exodus, a 5% workforce cut, collapsing share.
OLA = dict(
    leadership=[
        LeadershipEvent("Ola Electric", date(2026, 1, 19), Role.CFO,
                        "Harish Abichandani", EventType.RESIGNATION, verified=True),
        LeadershipEvent("Ola Electric", date(2025, 11, 12), Role.WHOLETIME_DIRECTOR,
                        "Suvonil Chatterjee (CTO)", EventType.RESIGNATION, verified=True),
        LeadershipEvent("Ola Electric", date(2025, 12, 3), Role.WHOLETIME_DIRECTOR,
                        "Anshul Khandelwal (CMO)", EventType.RESIGNATION, verified=True),
    ],
    headlines=[
        Headline(date(2026, 1, 20), "Ola Electric shares tank 5% as CFO Harish Abichandani resigns", "Business Today"),
        Headline(date(2026, 1, 20), "Ola Electric shares drop 7% as CFO resigns; stock slides 24% in 10 sessions", "Business Standard"),
        Headline(date(2026, 1, 31), "Ola Electric lays off 5% staff as EV sales stay below 10,000 units for third consecutive month", "Business Today"),
        Headline(date(2026, 5, 14), "Ola Electric market share slips further as rivals extend lead", "Economic Times"),
    ],
    hiring=[
        HiringObservation(date(2025, 6, 1), 12400),
        HiringObservation(date(2026, 1, 31), 11780),  # 5% cut, ~620 roles
        HiringObservation(date(2026, 5, 15), 11300),
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.1),
        RatingObservation(date(2026, 5, 15), 2.8),
    ],
)

# ============================================================== VODAFONE IDEA
# Negative net worth of ~Rs 35,800 crore behind a one-off accounting profit. Governance
# churn including the head of internal audit; new chairman brought in.
VODAFONE_IDEA = dict(
    leadership=[
        LeadershipEvent("Vodafone Idea", date(2026, 3, 6), Role.COMPANY_SECRETARY,
                        "Gautam Pendse (Head of Internal Audit)", EventType.RESIGNATION, verified=True),
        LeadershipEvent("Vodafone Idea", date(2026, 5, 5), Role.CHAIRMAN,
                        "Ravinder Takkar (stepped down as Chairman)", EventType.RESIGNATION, verified=True),
        LeadershipEvent("Vodafone Idea", date(2026, 6, 20), Role.WHOLETIME_DIRECTOR,
                        "Arvind Nevatia (Chief Enterprise Business Officer)", EventType.RESIGNATION, verified=True),
    ],
    headlines=[
        Headline(date(2026, 5, 5), "Kumar Mangalam Birla takes charge as Vodafone Idea chairman, Ravinder Takkar steps down", "People Matters"),
        Headline(date(2026, 5, 18), "Vodafone Idea appoints M P Sunil Kumar as Chief Enterprise Business Officer as Nevatia exits", "ScanX"),
        Headline(date(2026, 6, 15), "Vodafone Idea carries Rs 80,502 crore AGR liability; subscriber base continues to fall", "Business Standard"),
        Headline(date(2026, 6, 25), "Vodafone Idea trades on funding hopes as government support remains pivotal", "HDFC Sky"),
    ],
    hiring=[
        HiringObservation(date(2025, 6, 1), 9670),
        HiringObservation(date(2026, 5, 15), 9985),   # broadly flat headcount
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.9),
        RatingObservation(date(2026, 5, 15), 3.8),
    ],
)

# ===================================================================== VEDANTA
# The interesting one: levered financials (grey zone) but every market signal improving --
# four ratings upgrades, a completed demerger, and a top-100 workplace listing.
VEDANTA = dict(
    leadership=[
        LeadershipEvent("Vedanta", date(2026, 6, 1), Role.WHOLETIME_DIRECTOR,
                        "Arun Misra (tenure extended)", EventType.APPOINTMENT, verified=True),
    ],
    headlines=[
        Headline(date(2026, 4, 22), "Fitch upgrades Vedanta Resources to BB- from B+", "Fitch"),
        Headline(date(2026, 5, 14), "S&P Global upgrades Vedanta Resources to BB from B+", "S&P Global"),
        Headline(date(2026, 5, 28), "CRISIL upgrades Vedanta to AA+/Stable; ICRA follows after demerger clarity", "Business Today"),
        Headline(date(2026, 6, 27), "Vedanta ranks in India's top 100 best companies to work for; employees earn Rs 2,500 crore via ESOPs", "Asian Mirror"),
    ],
    hiring=[
        HiringObservation(date(2025, 6, 1), 16870),
        HiringObservation(date(2026, 5, 15), 16498),  # down ~2.2% year on year
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.4),
        RatingObservation(date(2026, 5, 15), 3.5),
    ],
)

# ======================================================================= PAYTM
# Recovery story: swung from a Rs 663 crore loss in FY25 to a Rs 552 crore profit in FY26,
# and is hiring again after two years of headcount reduction.
PAYTM = dict(
    leadership=[
        LeadershipEvent("Paytm", date(2026, 2, 10), Role.INDEPENDENT_DIRECTOR,
                        "board appointment", EventType.APPOINTMENT, verified=True),
    ],
    headlines=[
        Headline(date(2026, 4, 30), "Paytm posts third consecutive profitable quarter, revenue up 20% year on year", "Business Standard"),
        Headline(date(2026, 5, 21), "Paytm swings to Rs 552 crore profit in FY26 from Rs 663 crore loss", "Economic Times"),
        Headline(date(2026, 6, 12), "Paytm plans 4,000-person hiring spree for AI push", "Whalesbook"),
    ],
    hiring=[
        HiringObservation(date(2025, 3, 31), 39368),   # after a 10.4% reduction
        HiringObservation(date(2026, 5, 15), 41200),    # rehiring on the AI push
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.1),
        RatingObservation(date(2026, 5, 15), 3.2),
    ],
)

# ========================================================================= TCS
# Healthy financials, but the workforce signal is genuinely negative: headcount fell by
# ~23,500 in FY26 and attrition rose. The platform surfaces that without over-reacting.
TCS = dict(
    leadership=[
        LeadershipEvent("TCS", date(2025, 9, 1), Role.INDEPENDENT_DIRECTOR,
                        "board appointment", EventType.APPOINTMENT, verified=True),
    ],
    headlines=[
        Headline(date(2026, 4, 15), "TCS spends Rs 1,388 crore on restructuring in FY26 as 23,000 roles go, attrition climbs to 13.7%", "Storyboard18"),
        Headline(date(2026, 5, 6), "TCS HR head clarifies headcount drop, says on track to hire 40,000 freshers", "Storyboard18"),
        Headline(date(2026, 6, 18), "TCS rolls out 25,000 campus offers, says layoffs are over", "People Matters"),
    ],
    hiring=[
        HiringObservation(date(2025, 3, 31), 607979),
        HiringObservation(date(2026, 3, 31), 584510),  # down 3.85% year on year
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.7),
        RatingObservation(date(2026, 5, 15), 3.6),
    ],
)

_DATA = {
    "SpiceJet": SPICEJET,
    "Ola Electric": OLA,
    "Vodafone Idea": VODAFONE_IDEA,
    "Vedanta": VEDANTA,
    "Paytm": PAYTM,
    "TCS": TCS,
}

DEFAULT_AS_OF = {name: AS_OF for name in _DATA}


def has_signals(company: str) -> bool:
    return company in _DATA


def pulse_as_of(company: str, when: date, scorer: SentimentScorer | None = None) -> DigitalPulse:
    """Assemble the four-signal Digital Pulse for `company` as of `when`."""
    if company not in _DATA:
        raise KeyError(f"no signals for {company!r}; have {sorted(_DATA)}")
    d = _DATA[company]
    readings = [
        leadership_score(d["leadership"], when, company),
        sentiment_score(d["headlines"], when, scorer, company),
        hiring_score(d["hiring"], when, company),
        reviews_score(d["ratings"], when, company),
    ]
    return combine(company, when, readings)


def timeline(company: str, dates: list[date], scorer: SentimentScorer | None = None
             ) -> list[DigitalPulse]:
    """Digital Pulse at each date -- the trajectory for a case-study chart."""
    return [pulse_as_of(company, d, scorer) for d in dates]

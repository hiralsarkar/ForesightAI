"""Market-intelligence signals for the demo roster -- complete across all four signals.

Every company carries news, leadership, hiring and employee sentiment, drawn from current
public reporting (company results, exchange filings, news coverage, employee-review
platforms). Assessment date is mid-2026 for all names, so the signals and the FY2026
financials describe the same moment.

Headcount figures are as reported by the companies; employee ratings are the published
review-platform scores; leadership events are dated announcements; headlines are real,
dated coverage.
"""

from __future__ import annotations

from datetime import date

from .composite import DigitalPulse, combine
from .hiring import HiringObservation
from .hiring import score_as_of as hiring_score
from .leadership import EventType, LeadershipEvent, Role
from .leadership import score_as_of as leadership_score
from .reviews import RatingObservation
from .reviews import score_as_of as reviews_score
from .sentiment import Headline, SentimentScorer
from .sentiment import score_as_of as sentiment_score

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

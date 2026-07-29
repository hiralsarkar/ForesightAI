"""Module 3 -- Combined Risk Score: fusion of financial and digital signals.

The product's primary output. It fuses the Altman-anchored Financial Score with the
Digital Pulse composite into one 0-100 risk score, and -- crucially -- shows the two legs
separately, because the relationship *between* them is the story.

Two design decisions, both learned the hard way earlier in the build:

1. **Renormalize over available components; never score absent digital as 0.** The current
   roster carries digital signals on every company, but the fusion must stay correct for
   any company that lacks them: a fixed 60/40 with digital=0 would drag every
   financial-only company toward healthy (0.6*fin + 0.4*0) and under-flag a distressed
   one -- the same structural-missingness bias that broke serving in Phase 2. So when
   digital is absent, combined = financial, and the UI says so.

2. **No manufactured divergence.** The blueprint's flagship narrative is "the gap between
   the legs" (financial healthy, digital weak -> trouble not yet in the numbers). But all
   three dual-signal companies *agree* (TCS both healthy, Jet both critical, Future Retail
   both watch) -- there is no cross-sectional divergence in the real data, and fabricating
   one is exactly the mistake we corrected for Future Retail. So the narrative describes
   what is actually there: agreement (mutual confirmation) or, if it ever arises,
   divergence -- computed from the data, never assumed. The temporal thesis (digital
   leading) lives in the Jet timeline, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from ..serving.financial_score import FinancialScore, band_for
from ..signals.composite import DigitalPulse

DEFAULT_FINANCIAL_WEIGHT = 0.60
DEFAULT_DIGITAL_WEIGHT = 0.40

# Band order, worst last, for judging whether the two legs genuinely disagree. A gap
# within the same band (Jet: financial 100, digital 77 -- both Critical) is corroboration,
# not divergence; only a band-level disagreement is worth flagging.
_BAND_RANK = {"Healthy": 0, "Watch": 1, "Elevated Risk": 2, "Critical": 3}


@dataclass
class CombinedRisk:
    company: str
    as_of: Optional[date]
    financial_score: float
    digital_score: Optional[float]      # None when digital signals are unavailable
    combined_score: float
    financial_weight: float             # actual weight used (renormalized)
    digital_weight: float
    narrative: str

    @property
    def band(self) -> str:
        return band_for(self.combined_score)

    @property
    def has_digital(self) -> bool:
        return self.digital_score is not None

    @property
    def gap(self) -> Optional[float]:
        """digital - financial. Positive => digital is the weaker (riskier) leg."""
        if self.digital_score is None:
            return None
        return round(self.digital_score - self.financial_score, 1)


def _narrative(company: str, fin: float, dig: Optional[float], combined: float) -> str:
    band = band_for(combined).lower()

    if dig is None:
        return (
            f"{company} scores {combined:.0f}/100 ({band}) on financial signals alone. "
            "Digital signals are unavailable for this company, so the combined score "
            "reflects the financial model only."
        )

    fin_rank = _BAND_RANK[band_for(fin)]
    dig_rank = _BAND_RANK[band_for(dig)]

    if fin_rank == dig_rank:
        # The real case for every dual-signal demo company: the legs agree on the band.
        return (
            f"{company} scores {combined:.0f}/100 ({band}). Financial ({fin:.0f}) and "
            f"digital ({dig:.0f}) signals agree, which raises confidence in the assessment."
        )
    if dig_rank > fin_rank:
        # Digital reads a worse band than financials -- the blueprint's flagship pattern.
        # Computed from the data, never assumed; only fires if it genuinely occurs.
        return (
            f"{company} scores {combined:.0f}/100 ({band}), but digital signals "
            f"({band_for(dig).lower()}, {dig:.0f}) are weaker than the financials "
            f"({band_for(fin).lower()}, {fin:.0f}) suggest. This kind of divergence can "
            "precede financial deterioration -- a signal to look closer."
        )
    return (
        f"{company} scores {combined:.0f}/100 ({band}); the risk shows more in the "
        f"financials ({band_for(fin).lower()}, {fin:.0f}) than in market signals "
        f"({band_for(dig).lower()}, {dig:.0f}) so far."
    )


def fuse(
    financial: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    financial_weight: float = DEFAULT_FINANCIAL_WEIGHT,
    digital_weight: float = DEFAULT_DIGITAL_WEIGHT,
) -> CombinedRisk:
    """Combine the two legs into the primary risk score.

    When `digital` is None the combined score is the financial score and the weights
    collapse to (1.0, 0.0) -- no phantom digital=0 term.
    """
    fin_score = financial.risk_score

    if digital is None:
        combined = fin_score
        w_fin, w_dig, dig_score = 1.0, 0.0, None
    else:
        dig_score = digital.composite_score
        total = financial_weight + digital_weight
        w_fin, w_dig = financial_weight / total, digital_weight / total
        combined = w_fin * fin_score + w_dig * dig_score

    combined = round(combined, 1)
    return CombinedRisk(
        company=financial.company,
        as_of=digital.as_of if digital is not None else None,
        financial_score=round(fin_score, 1),
        digital_score=None if dig_score is None else round(dig_score, 1),
        combined_score=combined,
        financial_weight=round(w_fin, 2),
        digital_weight=round(w_dig, 2),
        narrative=_narrative(financial.company, fin_score, dig_score, combined),
    )

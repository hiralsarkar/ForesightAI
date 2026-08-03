"""Serving Financial Health Score, anchored on Altman Z''.

The serving work established that the GBM does not transfer to Indian companies while the
linear Altman Z'' does (see `docs/serving_indian_companies.md`). So the score a user sees for an Indian
company is derived from Z'', not from the GBM's calibrated probability.

Two things this module must get right:

1. **A 0-100 risk score monotonic in Z'', anchored on the published zone thresholds** so
   the bands line up exactly with Altman's own Distress / Grey / Safe boundaries. Higher
   score = higher risk (Z'' runs the other way: higher Z'' = safer).

2. **An exact 4-term decomposition** for the waterfall. Because Z'' is linear, each term
   is weight x value and the four terms *sum to the score* with no approximation -- a
   cleaner serving explanation than SHAP on a number we do not display.

The logistic anchoring solves for two fixed points:
    Z'' = 2.60 (safe threshold)     -> risk 25  (Healthy/Watch boundary)
    Z'' = 1.10 (distress threshold) -> risk 50  (Watch/Elevated boundary)
giving center z0 = 1.10 and steepness k = ln(3)/1.5 = 0.732. Everything below the
distress line lands >= 50 (Elevated/Critical); everything above the safe line lands < 25
(Healthy). This makes the gauge bands and the Altman zones the same statement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..features.altman import Z_DOUBLE_PRIME, ZSpec, get_spec
from ..features.polish_schema import label_for
from .screener import ScreenerFinancials, compute_features

# Logistic anchor constants (derived above; see module docstring).
_Z0 = 1.10
_K = math.log(3.0) / (Z_DOUBLE_PRIME.safe_above - Z_DOUBLE_PRIME.distress_below)  # 0.732

# Human labels for the four Z'' terms, in the language the waterfall shows.
_TERM_LABELS: dict[str, str] = {
    "Attr3": "Working Capital / Total Assets",
    "Attr6": "Retained Earnings / Total Assets",
    "Attr7": "Operating Return on Assets (EBIT/TA)",
    "Attr8": "Book Equity / Total Liabilities",
}


def risk_from_z(z: float) -> float:
    """Map an Altman Z'' to a 0-100 risk score (higher = riskier)."""
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return float("nan")
    return float(100.0 / (1.0 + math.exp(_K * (z - _Z0))))


def z_from_risk(risk: float) -> float:
    """Inverse map -- useful for slider bounds and reporting a target Z''."""
    risk = min(max(risk, 1e-6), 100 - 1e-6)
    return _Z0 + math.log(100.0 / risk - 1.0) / _K


@dataclass(frozen=True)
class ZTerm:
    """One term of the Z'' decomposition."""

    attr: str
    label: str
    coefficient: float
    value: float
    contribution: float  # coefficient * value; exact, sums to Z''

    @property
    def direction(self) -> str:
        # Every Z'' term is risk-*decreasing* in value (higher = safer), so a negative
        # contribution is what pushes toward distress.
        return "reduces risk" if self.contribution > 0 else "increases risk"


@dataclass
class FinancialScore:
    company: str
    year: int
    z_score: float
    risk_score: float          # 0-100
    band: str
    zone: str                  # Altman zone label
    terms: list[ZTerm]
    missing_terms: list[str]   # Z'' components that were unavailable (filled 0)

    @property
    def is_reliable(self) -> bool:
        """A score built with a missing high-weight term (WC/TA at 6.56) is softer."""
        return not self.missing_terms

    def top_drivers(self, n: int = 3) -> list[ZTerm]:
        """Terms pushing hardest toward distress, worst first."""
        return sorted(self.terms, key=lambda t: t.contribution)[:n]


# Bands must match calibrate.BANDS so financial and combined scores speak one language.
_BANDS: tuple[tuple[float, str], ...] = (
    (25.0, "Healthy"),
    (50.0, "Watch"),
    (75.0, "Elevated Risk"),
    (100.1, "Critical"),
)


def band_for(risk: float) -> str:
    if math.isnan(risk):
        return "Unknown"
    for upper, name in _BANDS:
        if risk < upper:
            return name
    return _BANDS[-1][1]


def score_company(
    fin: ScreenerFinancials,
    prior: ScreenerFinancials | None = None,
    spec: ZSpec = Z_DOUBLE_PRIME,
    fill_missing: bool = True,
) -> FinancialScore:
    """Compute the Altman-anchored financial score and its exact decomposition.

    `fill_missing=True` substitutes 0 for an unavailable Z'' component (a neutral
    assumption -- e.g. missing working capital -> WC/TA = 0) so the score still computes,
    and records it in `missing_terms`. With `fill_missing=False`, any missing component
    yields a NaN score, matching the strict training-path Altman behaviour.
    """
    from ..features.altman import zone as altman_zone

    feats = compute_features(fin, prior=prior)

    terms: list[ZTerm] = []
    missing: list[str] = []
    z = 0.0
    for attr, coef in spec.coefficients.items():
        val = feats.get(attr, float("nan"))
        if val is None or math.isnan(val):
            missing.append(_TERM_LABELS.get(attr, label_for(attr)))
            if not fill_missing:
                z = float("nan")
            val = 0.0
        contribution = coef * val
        if not math.isnan(z):
            z += contribution
        terms.append(
            ZTerm(attr, _TERM_LABELS.get(attr, label_for(attr)), coef, val, contribution)
        )

    if missing and not fill_missing:
        z = float("nan")

    risk = risk_from_z(z)
    import pandas as pd

    zone_label = (
        altman_zone(pd.Series([z]), "z2").iloc[0] if not math.isnan(z) else "Unknown"
    )

    return FinancialScore(
        company=fin.company,
        year=fin.year,
        z_score=z,
        risk_score=risk,
        band=band_for(risk),
        zone=str(zone_label),
        terms=terms,
        missing_terms=missing,
    )


def narrative(score: FinancialScore) -> str:
    """One-sentence plain-English summary of the financial score's drivers."""
    if math.isnan(score.risk_score):
        return f"{score.company} could not be scored: core balance-sheet terms were missing."

    band = score.band.lower()
    drivers = [t for t in score.top_drivers(3) if t.contribution < 0]
    if drivers:
        driver_text = ", ".join(t.label.split(" (")[0].lower() for t in drivers)
        tail = f"driven mainly by weak {driver_text}"
    else:
        strong = max(score.terms, key=lambda t: t.contribution)
        tail = f"supported by healthy {strong.label.split(' (')[0].lower()}"

    reliability = "" if score.is_reliable else " (note: working-capital data was unavailable)"
    return (
        f"{score.company} carries a {band} financial profile "
        f"(Altman Z'' {score.z_score:.1f}, risk {score.risk_score:.0f}/100), {tail}.{reliability}"
    )

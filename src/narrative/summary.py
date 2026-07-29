"""Module 6 -- AI Financial Narrative.

One paragraph, four sentences, in a credit analyst's voice. Two paths behind one call:

* **LLM path** (used when an API key is configured). The prompt receives *only
  pre-computed numbers* and is instructed never to calculate anything. This is the guard
  against hallucinated financials in a live demo -- the model narrates, it does not do
  arithmetic.
* **Deterministic fallback** (always available). Built from the same structured facts, so
  a missing API key or a timeout degrades the prose, never the correctness. AGENTS.md:
  "the demo must never break".

The fallback is written to vary in *substance*, not just in company name: which sentence
is chosen depends on the band, which Altman term dominates, and what the digital signals
say. Test it across companies -- if two outputs read the same, the templates are wrong.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

from ..scoring.combined import CombinedRisk
from ..serving.financial_score import FinancialScore
from ..signals.base import SignalKind
from ..signals.composite import DigitalPulse



@dataclass
class NarrativeFacts:
    """The structured input. Every number here is already computed -- the LLM only reads."""
    company: str
    sector: str
    combined_score: float
    combined_band: str
    financial_score: float
    financial_band: str
    altman_z: float
    altman_zone: str
    digital_score: Optional[float]
    digital_band: Optional[str]
    top_negative_terms: list[tuple[str, float]]   # (label, contribution) worst first
    top_positive_term: Optional[tuple[str, float]]
    signals: dict[str, str]                        # signal name -> its explanatory datum
    signals_available: bool


def collect_facts(
    combined: CombinedRisk,
    fin: FinancialScore,
    digital: Optional[DigitalPulse],
    sector: str = "",
) -> NarrativeFacts:
    negatives = [(t.label, t.contribution) for t in fin.terms if t.contribution < 0]
    negatives.sort(key=lambda x: x[1])
    positives = [(t.label, t.contribution) for t in fin.terms if t.contribution > 0]
    positives.sort(key=lambda x: -x[1])

    signals: dict[str, str] = {}
    if digital is not None:
        for r in digital.readings:
            signals[r.kind.value] = r.datum

    return NarrativeFacts(
        company=combined.company, sector=sector,
        combined_score=combined.combined_score, combined_band=combined.band,
        financial_score=fin.risk_score, financial_band=fin.band,
        altman_z=fin.z_score, altman_zone=fin.zone,
        digital_score=combined.digital_score,
        digital_band=None if digital is None else digital.band,
        top_negative_terms=negatives[:3], top_positive_term=positives[0] if positives else None,
        signals=signals, signals_available=digital is not None,
    )


# ------------------------------------------------------------------- LLM path
_SYSTEM = (
    "You are a senior credit analyst writing for a bank's risk committee. "
    "Write exactly four sentences in a single paragraph. No headings, no bullet points, "
    "no preamble, no hedging filler. "
    "CRITICAL: every number you need is supplied. Never calculate, infer, estimate or "
    "invent any figure, ratio, date or fact not present in the input. If something is not "
    "given, do not mention it. "
    "Identify the key risk drivers, note any divergence between financial and market "
    "signals, and close with what a lender should monitor going forward. "
    "Write plainly and specifically; do not sound like a chatbot or a template."
)


def _prompt(f: NarrativeFacts) -> str:
    lines = [
        f"Company: {f.company}" + (f" (sector: {f.sector})" if f.sector else ""),
        f"Combined risk score: {f.combined_score:.0f}/100 ({f.combined_band}); higher = riskier.",
        f"Financial score: {f.financial_score:.0f}/100 ({f.financial_band}).",
        f"Altman Z-double-prime: {f.altman_z:.2f} ({f.altman_zone} zone).",
    ]
    if f.digital_score is not None:
        lines.append(f"Digital/market-signal score: {f.digital_score:.0f}/100 ({f.digital_band}).")
    else:
        lines.append("Digital/market signals: NOT AVAILABLE for this company.")
    if f.top_negative_terms:
        drivers = "; ".join(f"{lbl} (contribution {c:+.2f})" for lbl, c in f.top_negative_terms)
        lines.append(f"Largest negative drivers of the financial score: {drivers}.")
    if f.top_positive_term:
        lines.append(f"Strongest supporting factor: {f.top_positive_term[0]} "
                     f"(contribution {f.top_positive_term[1]:+.2f}).")
    for name, datum in f.signals.items():
        lines.append(f"{name}: {datum}")
    return "\n".join(lines)


def _llm(f: NarrativeFacts, timeout: float = 15.0) -> Optional[str]:
    """Try the LLM via OpenRouter (OpenAI-compatible). Any failure returns None so the
    caller falls back silently -- the demo never breaks on an API problem.

    The prompt carries only pre-computed numbers and the system prompt forbids the model
    from calculating anything, so a weaker free model cannot invent a financial figure.
    """
    from .llm_config import OPENROUTER_URL, models, openrouter_key

    key = openrouter_key()
    if not key:
        return None

    import httpx

    payload_base = {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _prompt(f)},
        ],
        "max_tokens": 400,
        "temperature": 0.4,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    for model in models():  # try the preferred model, then fallbacks
        try:
            resp = httpx.post(OPENROUTER_URL, headers=headers,
                              json={**payload_base, "model": model}, timeout=timeout)
            if resp.status_code != 200:
                continue
            text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            if text:
                return text
        except Exception:
            continue
    return None


# ------------------------------------------------------------ deterministic path
def _band_clause(band: str) -> str:
    return {
        "Healthy": "sits comfortably outside the distress range",
        "Watch": "belongs on the watchlist rather than in the problem book",
        "Elevated Risk": "warrants active management attention now",
        "Critical": "is in genuine financial distress",
    }.get(band, "has been assessed")


def _fallback(f: NarrativeFacts) -> str:
    # 1. Position.
    s1 = (f"{f.company} scores {f.combined_score:.0f} out of 100 ({f.combined_band.lower()}) "
          f"and {_band_clause(f.combined_band)}, with an Altman Z-double-prime of "
          f"{f.altman_z:.1f} placing it in the {f.altman_zone.lower()} zone.")

    # 2. What drives it -- differs by whether the profile is weak or strong.
    if f.top_negative_terms:
        names = [lbl.split(" (")[0].lower() for lbl, _ in f.top_negative_terms]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
        s2 = f"The score is driven primarily by weak {joined}."
    elif f.top_positive_term:
        s2 = (f"No individual balance-sheet term is pulling the score down, with "
              f"{f.top_positive_term[0].split(' (')[0].lower()} the strongest support.")
    else:
        s2 = "No individual balance-sheet term dominates the assessment."

    # 3. Market signals -- corroboration, divergence, or absence.
    if not f.signals_available:
        s3 = ("Market-intelligence signals are not available for this company, so the "
              "assessment rests on reported financials alone.")
    else:
        # Use the SAME band-disagreement test as the Module 3 combined narrative. Keying
        # this off a numeric gap instead let the two panels contradict each other on
        # screen -- one calling it a divergence while the other called it corroboration.
        from ..scoring.combined import _BAND_RANK

        fin_rank = _BAND_RANK.get(f.financial_band, 0)
        dig_rank = _BAND_RANK.get(f.digital_band or f.financial_band, 0)
        concern = max(f.signals.items(), key=lambda kv: len(kv[1])) if f.signals else None
        if dig_rank > fin_rank:
            s3 = (f"Market signals are weaker than the financials suggest "
                  f"({f.digital_band.lower()} at {f.digital_score:.0f} against a "
                  f"{f.financial_band.lower()} {f.financial_score:.0f})"
                  + (f"; {concern[0].lower()} is the sharpest of them." if concern else "."))
        elif dig_rank < fin_rank:
            s3 = (f"Market signals ({f.digital_score:.0f}) read better than the financials "
                  f"({f.financial_score:.0f}), so the strain is currently visible in the "
                  "numbers more than in external indicators.")
        else:
            s3 = (f"Market signals agree with the financial picture at "
                  f"{f.digital_score:.0f}, which raises confidence in the assessment.")

    # 4. What to monitor -- band-specific, actionable.
    s4 = {
        "Healthy": "A lender should maintain the standard review cycle and watch for "
                   "debt-funded expansion rather than trading deterioration.",
        "Watch": "A lender should shorten the reporting cycle and re-test covenant headroom "
                 "before the next renewal.",
        "Elevated Risk": "A lender should request updated management accounts now and "
                         "reassess security cover rather than waiting for the next filing.",
        "Critical": "A lender should treat further drawdowns as refinancing, verify "
                    "collateral independently, and prepare for restructuring discussions.",
    }.get(f.combined_band, "A lender should keep this name under periodic review.")

    return " ".join([s1, s2, s3, s4])


# ---------------------------------------------------------------- narrative cache
# Pre-generated LLM narratives, committed to the repo so the demo works OFFLINE. The key
# is a hash of the exact prompt, so if a company's numbers change the cache misses and the
# text regenerates (or falls back) rather than showing stale prose. This is genuine LLM
# output, just persisted -- so a cache hit is still reported as 'llm'.
import hashlib
import json as _json
from pathlib import Path as _Path

_CACHE_PATH = _Path(__file__).resolve().parents[2] / "data" / "demo" / "narrative_cache.json"


def _cache_key(f: NarrativeFacts) -> str:
    return hashlib.sha256(_prompt(f).encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict:
    try:
        return _json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _store_cache(key: str, text: str) -> None:
    cache = _load_cache()
    cache[key] = text
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(_json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# -------------------------------------------------------------------- public API
def generate(
    combined: CombinedRisk,
    fin: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    sector: str = "",
    use_llm: bool = True,
    use_cache: bool = True,
) -> tuple[str, str]:
    """Return `(narrative, source)` where source is 'llm' or 'rule-based'.

    Resolution order: pre-generated cache -> live LLM (and cache the result) -> rule-based.
    Never raises and never depends on the network when the cache is warm, which is what
    makes the live demo safe offline.
    """
    facts = collect_facts(combined, fin, digital, sector)
    if use_llm:
        key = _cache_key(facts)
        if use_cache:
            cached = _load_cache().get(key)
            if cached:
                return cached, "llm"
        text = _llm(facts)
        if text:
            _store_cache(key, text)
            return text, "llm"
    return _fallback(facts), "rule-based"


def prewarm_cache(quiet: bool = False) -> int:
    """Populate the narrative cache for the whole demo roster (run this online).

    Uses the same FinBERT-scored path the app uses, so the cache keys match exactly what
    the live dashboard will request. Returns the number of narratives cached.
    """
    from ..scoring.combined import fuse
    from ..serving.demo_companies import ROSTER, SECTORS
    from ..serving.financial_score import score_company
    from ..signals.demo_signals import DEFAULT_AS_OF, pulse_as_of
    from ..signals.sentiment import default_scorer

    scorer = default_scorer()  # FinBERT if available -- must match the app
    cache = _load_cache()
    n = 0
    for rec, prior, _ in ROSTER:
        fin = score_company(rec, prior=prior)
        dig = pulse_as_of(rec.company, DEFAULT_AS_OF[rec.company], scorer)
        sector = SECTORS.get(rec.company, "")  # MUST match the app's call
        combined = fuse(fin, dig)
        key = _cache_key(collect_facts(combined, fin, dig, sector))
        if key in cache:  # idempotent -- re-runs only fill gaps, sparing the rate limit
            n += 1
            if not quiet:
                print(f"  already cached: {rec.company}")
            continue
        text, src = generate(combined, fin, dig, sector=sector, use_cache=False)
        if src == "llm":
            n += 1
            if not quiet:
                print(f"  cached: {rec.company}")
        elif not quiet:
            print(f"  FALLBACK (not cached, retry): {rec.company}")
    return n

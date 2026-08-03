from __future__ import annotations

"""LLM provider configuration for the narrative.

The key is read from the environment first, then from a gitignored `secrets.local.json`
at the repo root. It is never hard-coded in committed source (see `.gitignore`).
"""


import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# OpenRouter is OpenAI-compatible. A free instruction-following model is the default;
# override with the OPENROUTER_MODEL env var if a stronger one is available on the key.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# gpt-oss-20b returns clean four-sentence prose; the Nemotron models leak their
# chain-of-thought ("We need to craft...") so they are not used. Gemma is a rate-limited
# fallback. Verified against the live key on 2026-07-21.
DEFAULT_MODEL = "openai/gpt-oss-20b:free"
_FALLBACK_MODELS = (
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
)


def openrouter_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    f = _ROOT / "secrets.local.json"
    if f.exists():
        try:
            return (json.loads(f.read_text(encoding="utf-8")).get("openrouter_api_key") or "").strip() or None
        except Exception:
            return None
    return None


def models() -> tuple[str, ...]:
    override = os.environ.get("OPENROUTER_MODEL")
    if override:
        return (override, *(_m for _m in _FALLBACK_MODELS if _m != override))
    return _FALLBACK_MODELS


def is_configured() -> bool:
    return openrouter_key() is not None

"""AI Financial Narrative.

One paragraph, four sentences, in a credit analyst's voice. Two paths behind one call:

* **LLM path** (used when an API key is configured). The prompt receives *only
  pre-computed numbers* and is instructed never to calculate anything. This is the guard
  against hallucinated financials in a live demo -- the model narrates, it does not do
  arithmetic.
* **Deterministic fallback** (always available). Built from the same structured facts, so
  a missing API key or a timeout degrades the prose, never the correctness. By design:
  "the demo keeps working".

The fallback is written to vary in *substance*, not just in company name: which sentence
is chosen depends on the band, which Altman term dominates, and what the digital signals
say. Test it across companies -- if two outputs read the same, the templates are wrong.
"""


import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

from src.scoring import CombinedRisk
from src.scoring import FinancialScore
from src.signals import SignalKind
from src.signals import DigitalPulse



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
    caller falls back silently -- an API problem never degrades the output.

    The prompt carries only pre-computed numbers and the system prompt forbids the model
    from calculating anything, so a weaker free model cannot invent a financial figure.
    """

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
        # Use the SAME band-disagreement test as the combined narrative. Keying
        # this off a numeric gap instead let the two panels contradict each other on
        # screen -- one calling it a divergence while the other called it corroboration.
        from src.scoring import _BAND_RANK

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

_CACHE_PATH = _Path(__file__).resolve().parents[1] / "data" / "demo" / "narrative_cache.json"


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
    from src.scoring import fuse
    from src.scoring import ROSTER, SECTORS
    from src.scoring import score_company
    from src.signals import DEFAULT_AS_OF, pulse_as_of
    from src.signals import default_scorer

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

"""Risk Mitigation Recommendations.

**Rule-based by design, never LLM-generated** (rule-based by design). These sit next to a number
a credit committee may act on, so they must be reproducible and incapable of inventing a
fact. Every rule is tied to actual metric values, not generic advice.

Output is split by audience, which is what shows the platform serves multiple buyers:
  * FOR LENDERS / INVESTORS -- what to do if you have exposure.
  * FOR MANAGEMENT -- what to do if this is your own company.

Rules fire independently and are ranked by priority, so a company can trigger several.
"""


import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from src.scoring import FinancialScore
from src.scoring import ScreenerFinancials, compute_features
from src.signals import SignalKind
from src.signals import DigitalPulse


class Audience(str, Enum):
    LENDER = "For Lenders / Investors"
    MANAGEMENT = "For Management"


@dataclass(frozen=True)
class Recommendation:
    audience: Audience
    title: str
    action: str
    priority: int  # 1 = most urgent


@dataclass
class _Ctx:
    """Everything the rules read. Missing values are NaN/None and rules must tolerate it."""
    company: str
    fin: FinancialScore
    interest_cover: float
    #: total liabilities / assets -- INCLUDES trade payables and provisions, so this is
    #: not "debt". FMCG names run high here on supplier credit while carrying no borrowing.
    debt_to_assets: float
    #: interest-bearing borrowings / assets. Use THIS for any statement about leverage or
    #: debt, never `debt_to_assets`, or debt-free companies get told they are levered.
    borrowings_to_assets: float
    equity_ratio: float
    roa: float
    op_margin: float
    wc_to_assets: float
    digital: Optional[DigitalPulse]

    def sig(self, kind: SignalKind) -> Optional[float]:
        if self.digital is None:
            return None
        r = self.digital.by_kind(kind)
        return None if r is None else r.risk_score

    def sig_label(self, kind: SignalKind) -> str:
        if self.digital is None:
            return ""
        r = self.digital.by_kind(kind)
        return "" if r is None else r.label

    def sig_raw(self, kind: SignalKind) -> Optional[float]:
        if self.digital is None:
            return None
        r = self.digital.by_kind(kind)
        return None if r is None else r.raw


def _ok(x: Optional[float]) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


Rule = Callable[[_Ctx], Optional[Recommendation]]
_RULES: list[Rule] = []


def _rule(fn: Rule) -> Rule:
    _RULES.append(fn)
    return fn


# ------------------------------------------------------------------ lender rules
@_rule
def _r_negative_equity(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.equity_ratio) and c.equity_ratio < 0:
        return Recommendation(
            Audience.LENDER, "Negative net worth",
            f"Book equity is negative ({c.equity_ratio:.0%} of assets): liabilities exceed "
            "assets, so unsecured exposure has no equity cushion behind it. Escalate to "
            "special-mention, re-verify collateral values and enforce security where held.", 1)
    return None


@_rule
def _r_cannot_service_debt(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and c.interest_cover < 1.0:
        return Recommendation(
            Audience.LENDER, "Interest not covered by operations",
            f"Interest coverage is {c.interest_cover:.2f}x - operating profit does not cover "
            "the interest bill, so debt service is being funded from reserves or new "
            "borrowing. Treat further drawdown requests as refinancing, not growth.", 1)
    return None


@_rule
def _r_thin_cover_high_leverage(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and _ok(c.debt_to_assets) and 1.0 <= c.interest_cover < 2.0 and c.debt_to_assets > 0.65:
        return Recommendation(
            Audience.LENDER, "Debt servicing capacity is thin relative to leverage",
            f"Coverage of {c.interest_cover:.2f}x against liabilities at {c.debt_to_assets:.0%} "
            "of assets leaves little headroom for a rate or earnings shock. Reassess covenant "
            "levels and consider tightening reporting frequency to monthly.", 2)
    return None


@_rule
def _r_distress_zone(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.zone == "Distress":
        return Recommendation(
            Audience.LENDER, "Altman Z-Score in the distress zone",
            f"Z'' of {c.fin.z_score:.2f} sits below the 1.10 distress threshold. Move the "
            "name to the watchlist, refresh the internal rating, and require updated "
            "management accounts before any limit renewal.", 2)
    return None


@_rule
def _r_loss_making(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.roa) and c.roa < 0:
        return Recommendation(
            Audience.LENDER, "Loss-making at the asset level",
            f"Return on assets is {c.roa:.0%}: the asset base is destroying capital rather "
            "than generating it. Confirm whether losses are one-off or structural before "
            "extending the facility.", 2)
    return None


@_rule
def _r_negative_working_capital(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.wc_to_assets) and c.wc_to_assets < -0.10 and _ok(c.interest_cover) and c.interest_cover < 2.0:
        return Recommendation(
            Audience.LENDER, "Negative working capital alongside weak coverage",
            f"Working capital is {c.wc_to_assets:.0%} of assets while coverage is only "
            f"{c.interest_cover:.2f}x. Short-term obligations are being met from operating "
            "inflows with no buffer - a single collection delay could trigger default.", 2)
    return None


@_rule
def _r_leadership_exodus(c: _Ctx) -> Optional[Recommendation]:
    n = c.sig_raw(SignalKind.LEADERSHIP)
    if n is not None and n > 2:
        return Recommendation(
            Audience.LENDER, "Senior leadership turnover",
            f"{int(n)} senior departures filed in the last six months. Concentrated exits at "
            "board or KMP level often precede disclosure events; request a governance update "
            "and confirm who now holds financial authority.", 2)
    return None


@_rule
def _r_operational_stress(c: _Ctx) -> Optional[Recommendation]:
    hire = c.sig_label(SignalKind.HIRING)
    sent = c.sig(SignalKind.NEWS_SENTIMENT)
    if hire == "Contracting" and sent is not None and sent >= 50:
        return Recommendation(
            Audience.LENDER, "Workforce contraction with negative coverage",
            "Hiring is contracting while news sentiment is negative - a pattern that "
            "indicates operational stress not yet reflected in the reported financials. "
            "Request updated management accounts rather than waiting for the next filing.", 2)
    return None


@_rule
def _r_digital_worse_than_financial(c: _Ctx) -> Optional[Recommendation]:
    if c.digital is None:
        return None
    gap = c.digital.composite_score - c.fin.risk_score
    if gap > 15:
        return Recommendation(
            Audience.LENDER, "Market signals weaker than the financials suggest",
            f"Digital signals score {c.digital.composite_score:.0f} against a financial score "
            f"of {c.fin.risk_score:.0f}. Where market signals lead the statements, the next "
            "reporting period often confirms them - bring forward the review date.", 3)
    return None


@_rule
def _r_no_digital_coverage(c: _Ctx) -> Optional[Recommendation]:
    if c.digital is None:
        return Recommendation(
            Audience.LENDER, "No market-intelligence coverage",
            "This assessment rests on reported financials alone. Establish news, hiring and "
            "filing monitoring for this name so deterioration between reporting dates is "
            "visible.", 4)
    return None


@_rule
def _r_grey_zone_watch(c: _Ctx) -> Optional[Recommendation]:
    """The watchlist band is where a lender most needs guidance -- without this, a company
    that is neither clearly safe nor clearly distressed produces no lender action."""
    if c.fin.zone == "Grey" or c.fin.band == "Watch":
        lev = (f" with liabilities at {c.debt_to_assets:.0%} of assets"
               if _ok(c.debt_to_assets) and c.debt_to_assets > 0.6 else "")
        return Recommendation(
            Audience.LENDER, "Grey zone - neither clearly safe nor distressed",
            f"Z'' of {c.fin.z_score:.2f} falls between the distress and safe thresholds{lev}. "
            "This is the band where outcomes diverge most: move to semi-annual review, "
            "confirm covenant headroom, and track the market signals for early deterioration.", 3)
    return None


@_rule
def _r_leverage_concentration(c: _Ctx) -> Optional[Recommendation]:
    # Gated on actual borrowings: a company funded by supplier credit is not "levered".
    if _ok(c.borrowings_to_assets) and c.borrowings_to_assets >= 0.30:
        return Recommendation(
            Audience.LENDER, "High balance-sheet leverage",
            f"Interest-bearing borrowings fund {c.borrowings_to_assets:.0%} of the asset base, "
            "so a modest fall in asset values would erode the equity cushion. Re-test security "
            "cover against current, not historic, valuations.", 3)
    return None


@_rule
def _r_healthy_but_levered(c: _Ctx) -> Optional[Recommendation]:
    if (c.fin.band == "Healthy" and _ok(c.borrowings_to_assets) and c.borrowings_to_assets > 0.20
            and _ok(c.interest_cover) and c.interest_cover >= 2.0):
        return Recommendation(
            Audience.LENDER, "Sound today, but leverage is material",
            f"Borrowings are {c.borrowings_to_assets:.0%} of assets even though coverage is "
            f"comfortable at {c.interest_cover:.1f}x. Monitor covenant headroom under a rate "
            "rise; the profile is sound but not shock-proof.", 4)
    return None


@_rule
def _r_healthy_standard_cycle(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.band == "Healthy" and (c.digital is None or c.digital.band == "Healthy"):
        return Recommendation(
            Audience.LENDER, "No action indicated",
            f"Financial and market signals are both in the healthy band (Z'' "
            f"{c.fin.z_score:.1f}). Maintain the standard annual review cycle; no additional "
            "monitoring is warranted on current evidence.", 5)
    return None


# -------------------------------------------------------------- management rules
@_rule
def _r_mgmt_recapitalize(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.equity_ratio) and c.equity_ratio < 0:
        return Recommendation(
            Audience.MANAGEMENT, "Restore the equity base",
            "Net worth is negative, which constrains new borrowing and may breach covenants. "
            "Priority is a recapitalisation - rights issue, promoter infusion, or converting "
            "debt to equity - before refinancing terms deteriorate further.", 1)
    return None


@_rule
def _r_mgmt_refinance(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and c.interest_cover < 1.5:
        return Recommendation(
            Audience.MANAGEMENT, "Reduce the debt service burden",
            f"At {c.interest_cover:.2f}x coverage, interest is consuming most or all of "
            "operating profit. Open refinancing discussions early - extending maturities and "
            "resetting covenants is far cheaper while the company is still performing.", 1)
    return None


@_rule
def _r_mgmt_working_capital(c: _Ctx) -> Optional[Recommendation]:
    # Negative working capital is a *strength* in FMCG and retail -- suppliers fund the
    # business. It is only a problem when the company is not otherwise healthy, so this
    # rule is gated on the band. Telling a debt-free FMCG name to "close the gap" would be
    # wrong advice and would cost credibility instantly.
    if c.fin.band == "Healthy":
        return None
    if _ok(c.wc_to_assets) and c.wc_to_assets < -0.10:
        return Recommendation(
            Audience.MANAGEMENT, "Close the working capital gap",
            f"Working capital is {c.wc_to_assets:.0%} of assets. Negotiate a committed "
            "working-capital facility and tighten receivables collection so operations are "
            "not dependent on continuous supplier financing.", 2)
    return None


@_rule
def _r_mgmt_margins(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.op_margin) and c.op_margin < 0.05:
        return Recommendation(
            Audience.MANAGEMENT, "Rebuild operating margin",
            f"Operating margin is {c.op_margin:.1%}, leaving no absorption for input-cost or "
            "demand shocks. A structural cost review will do more for the risk profile than "
            "further balance-sheet engineering.", 2)
    return None


@_rule
def _r_mgmt_leadership(c: _Ctx) -> Optional[Recommendation]:
    n = c.sig_raw(SignalKind.LEADERSHIP)
    if n is not None and n >= 2:
        return Recommendation(
            Audience.MANAGEMENT, "Stabilise the leadership team",
            f"{int(n)} senior departures in six months is visible to lenders and rating "
            "agencies. Publish a clear succession and retention plan; unexplained churn is "
            "read as a governance signal regardless of the underlying reason.", 2)
    return None


@_rule
def _r_mgmt_workforce(c: _Ctx) -> Optional[Recommendation]:
    if c.sig_label(SignalKind.HIRING) == "Contracting" or c.sig_label(SignalKind.EMPLOYEE) == "Declining":
        return Recommendation(
            Audience.MANAGEMENT, "Address workforce signals",
            "Hiring contraction and falling employee sentiment are externally visible and "
            "feed third-party risk models. Communicate the operating plan internally before "
            "attrition compounds the problem.", 3)
    return None


@_rule
def _r_mgmt_communication(c: _Ctx) -> Optional[Recommendation]:
    s = c.sig(SignalKind.NEWS_SENTIMENT)
    if s is not None and s >= 55:
        return Recommendation(
            Audience.MANAGEMENT, "Get ahead of the narrative",
            "Recent coverage is materially negative. Proactive disclosure on liquidity and "
            "the funding plan is more effective than silence - lenders price uncertainty "
            "more harshly than bad news.", 3)
    return None


@_rule
def _r_mgmt_maintain(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.band == "Healthy":
        return Recommendation(
            Audience.MANAGEMENT, "Preserve the current position",
            f"The balance sheet is sound (Z'' {c.fin.z_score:.1f}). Maintain the current "
            "leverage discipline; the main risk to this profile is debt-funded expansion "
            "rather than trading performance.", 5)
    return None


# -------------------------------------------------------------------- public API
def build_context(
    rec: ScreenerFinancials,
    fin: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    prior: Optional[ScreenerFinancials] = None,
) -> _Ctx:
    f = compute_features(rec, prior=prior)
    bta = (rec.borrowings / rec.total_assets) if rec.total_assets else float("nan")
    return _Ctx(
        company=rec.company, fin=fin,
        interest_cover=f.get("Attr27", float("nan")),
        debt_to_assets=f.get("Attr2", float("nan")),
        borrowings_to_assets=bta,
        equity_ratio=f.get("Attr10", float("nan")),
        roa=f.get("Attr1", float("nan")),
        op_margin=f.get("Attr42", float("nan")),
        wc_to_assets=f.get("Attr3", float("nan")),
        digital=digital,
    )


def recommend(
    rec: ScreenerFinancials,
    fin: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    prior: Optional[ScreenerFinancials] = None,
) -> dict[Audience, list[Recommendation]]:
    """Fire every rule and group the hits by audience, most urgent first."""
    ctx = build_context(rec, fin, digital, prior)
    out: dict[Audience, list[Recommendation]] = {Audience.LENDER: [], Audience.MANAGEMENT: []}
    for rule in _RULES:
        r = rule(ctx)
        if r is not None:
            out[r.audience].append(r)
    for aud in out:
        out[aud].sort(key=lambda r: r.priority)
    return out


def rule_count() -> int:
    return len(_RULES)

"""Executive Report: a two-page credit-intelligence memo (ReportLab).

Page 1  identity, combined risk score, key financial metrics, market signals, narrative.
Page 2  Altman decomposition, recommendations by audience, disclaimer.

Two deliberate choices on page 2, both evidence-driven:

* No SHAP waterfall. Our *serving* explanation is the exact Altman Z'' 4-term
  decomposition (the GBM does not drive the displayed score), so that is what we print.
* No debt/profitability *trajectory* charts. We curated a single fiscal year
  for seven of eight companies, so multi-year trajectories do not exist for them and are
  not invented. Where a prior year exists the year-on-year move is printed as a line of
  text instead.

The closing disclaimer is not weakness -- real enterprise risk reporting carries one.
"""


from datetime import datetime
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

from src.scoring import CombinedRisk
from src.scoring import FinancialScore
from src.scoring import ScreenerFinancials, compute_features
from src.signals import DigitalPulse

# Palette mirrors the dashboard so the report is recognisably the same product.
NAVY = colors.HexColor("#0A1628")
PANEL = colors.HexColor("#0F1E33")
AMBER = colors.HexColor("#F59E0B")
DIM = colors.HexColor("#94A3B8")
BORDER = colors.HexColor("#1E3350")
GOOD = colors.HexColor("#22C55E")
WATCH = colors.HexColor("#F59E0B")
ELEVATED = colors.HexColor("#F97316")
BAD = colors.HexColor("#EF4444")

_BAND_COLOR = {"Healthy": GOOD, "Watch": WATCH, "Elevated Risk": ELEVATED, "Critical": BAD}

_PAGE_W, _PAGE_H = A4
_MARGIN = 16 * mm
_BANNER_H = 22 * mm


def _band_color(band: str):
    return _BAND_COLOR.get(band, DIM)


# ------------------------------------------------------------------------ styles
def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9, leading=13,
                          textColor=colors.HexColor("#1F2937"), alignment=TA_LEFT)
    return {
        "body": base,
        "section": ParagraphStyle("section", parent=base, fontName="Helvetica-Bold",
                                  fontSize=8, textColor=colors.HexColor("#64748B"),
                                  spaceAfter=4, leading=11),
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=17,
                             textColor=NAVY, leading=20),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9, textColor=colors.HexColor("#64748B")),
        "narr": ParagraphStyle("narr", parent=base, fontSize=9.5, leading=14),
        "rec_t": ParagraphStyle("rec_t", parent=base, fontName="Helvetica-Bold", fontSize=8.5,
                                leading=11),
        "rec_b": ParagraphStyle("rec_b", parent=base, fontSize=8, leading=11,
                                textColor=colors.HexColor("#475569")),
        "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=DIM),
        "disc": ParagraphStyle("disc", parent=base, fontSize=7.5, leading=10,
                               textColor=colors.HexColor("#64748B")),
    }


# ------------------------------------------------------------- page furniture
def _draw_page(canvas, doc, company: str, generated: str) -> None:
    canvas.saveState()
    # Navy banner with the brand.
    canvas.setFillColor(NAVY)
    canvas.rect(0, _PAGE_H - _BANNER_H, _PAGE_W, _BANNER_H, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(_MARGIN, _PAGE_H - 14 * mm, "Foresight")
    canvas.setFillColor(AMBER)
    canvas.drawString(_MARGIN + 24 * mm, _PAGE_H - 14 * mm, "AI")
    canvas.setFillColor(DIM)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(_PAGE_W - _MARGIN, _PAGE_H - 14 * mm,
                           "CREDIT INTELLIGENCE MEMO")
    # Footer.
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(_MARGIN, 10 * mm, f"{company}  |  Generated {generated}")
    canvas.drawRightString(_PAGE_W - _MARGIN, 10 * mm, f"Page {canvas.getPageNumber()} of 2")
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.line(_MARGIN, 13 * mm, _PAGE_W - _MARGIN, 13 * mm)
    canvas.restoreState()


# ------------------------------------------------------------------- components
def _score_block(combined: CombinedRisk, fin: FinancialScore, st) -> Table:
    """Large coloured score box plus the two component legs."""
    c = _band_color(combined.band)
    score_cell = Table(
        [[Paragraph(f'<font size="30" color="white"><b>{combined.combined_score:.0f}</b></font>'
                    f'<font size="11" color="white">/100</font>', st["body"])],
         [Paragraph(f'<font size="10" color="white"><b>{combined.band.upper()}</b></font>', st["body"])]],
        colWidths=[46 * mm], rowHeights=[16 * mm, 7 * mm])
    score_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    dig = f"{combined.digital_score:.0f}/100" if combined.has_digital else "Not available"
    detail = Table([
        ["Financial Health", f"{combined.financial_score:.0f}/100",
         f"weight {combined.financial_weight:.0%}"],
        ["Market Signals", dig, f"weight {combined.digital_weight:.0%}"],
        ["Altman Z''", f"{fin.z_score:.2f}", f"{fin.zone} zone"],
    ], colWidths=[34 * mm, 26 * mm, 32 * mm])
    detail.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("FONT", (1, 0), (1, -1), "Helvetica-Bold", 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#94A3B8")),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    wrap = Table([[score_cell, detail]], colWidths=[50 * mm, 96 * mm])
    wrap.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (0, 0), 0)]))
    return wrap


def _metrics_table(rec: ScreenerFinancials, prior, st) -> Table:
    f = compute_features(rec, prior=prior)

    def fmt(key, pct=False, suffix=""):
        v = f.get(key, float("nan"))
        if v != v:
            return "n/a", DIM
        txt = f"{v*100:.0f}%" if pct else f"{v:.2f}{suffix}"
        return txt, None

    rows = [
        ("Interest Coverage", *fmt("Attr27", suffix="x"), lambda v: v >= 2, "Attr27"),
        ("Debt to Assets", *fmt("Attr2", pct=True), lambda v: v < 0.7, "Attr2"),
        ("Return on Assets", *fmt("Attr1", pct=True), lambda v: v > 0, "Attr1"),
        ("Equity Ratio", *fmt("Attr10", pct=True), lambda v: v > 0.2, "Attr10"),
    ]
    data, styles = [[], []], []
    for i, (label, txt, _dim, ok, key) in enumerate(rows):
        v = f.get(key, float("nan"))
        col = DIM if v != v else (GOOD if ok(v) else BAD)
        data[0].append(Paragraph(f'<font size="7" color="#64748B">{label.upper()}</font>', st["body"]))
        data[1].append(Paragraph(f'<font size="13" color="{col.hexval()}"><b>{txt}</b></font>', st["body"]))
    t = Table(data, colWidths=[36.5 * mm] * 4)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _signals_block(digital: Optional[DigitalPulse], st) -> list:
    if digital is None:
        return [Paragraph("Market-intelligence signals are not available for this company; "
                          "the assessment rests on reported financials alone.", st["body"])]
    out = []
    for r in digital.readings:
        col = _band_color(r.band)
        out.append(Paragraph(
            f'<font color="{col.hexval()}">•</font> <b>{r.kind.value}</b> '
            f'<font color="#64748B">({r.label})</font> - {r.datum}', st["body"]))
        out.append(Spacer(1, 2.5 * mm))
    return out


def _decomposition(fin: FinancialScore, st) -> Table:
    """Exact Altman Z'' 4-term decomposition -- the terms sum to the score."""
    maxc = max((abs(t.contribution) for t in fin.terms), default=1.0) or 1.0
    data = []
    for t in fin.terms:
        col = GOOD if t.contribution > 0 else BAD
        width = max(1.0, abs(t.contribution) / maxc * 42)
        bar = Table([[""]], colWidths=[width * mm], rowHeights=[3.4 * mm])
        bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        data.append([
            Paragraph(f'<font size="8">{t.label}</font>', st["body"]), bar,
            Paragraph(f'<font size="8" color="{col.hexval()}"><b>{t.contribution:+.2f}</b></font>'
                      f'<font size="7" color="#94A3B8">  ({t.value:+.2f} &times; {t.coefficient})</font>',
                      st["body"]),
        ])
    data.append([Paragraph('<font size="8"><b>Altman Z'' (sum)</b></font>', st["body"]), "",
                 Paragraph(f'<font size="8"><b>{fin.z_score:+.2f}</b></font>', st["body"])])
    t = Table(data, colWidths=[62 * mm, 44 * mm, 40 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    return t


def _recs_block(recs: dict[Audience, list[Recommendation]], st) -> list:
    out = []
    for aud in (Audience.LENDER, Audience.MANAGEMENT):
        items = recs.get(aud, [])
        out.append(Paragraph(f'<font color="#B45309"><b>{aud.value.upper()}</b></font>', st["section"]))
        out.append(Spacer(1, 1.5 * mm))
        if not items:
            out.append(Paragraph("No specific actions indicated.", st["rec_b"]))
        for it in items[:4]:  # keep the memo to two pages
            out.append(Paragraph(f"{it.title}", st["rec_t"]))
            out.append(Paragraph(it.action, st["rec_b"]))
            out.append(Spacer(1, 2 * mm))
        out.append(Spacer(1, 3 * mm))
    return out


def _yoy_line(rec: ScreenerFinancials, prior: Optional[ScreenerFinancials], st):
    """Year-on-year move where a prior year exists; otherwise say so plainly."""
    if prior is None:
        return Paragraph("Only a single reported year was available for this company, so a "
                         "multi-year trajectory is not shown.", st["small"])

    def move(now, then):
        """Percentage change, but fall back to absolute values when a percentage would
        mislead. Jet's operating profit went +24 -> -3,660: '-15350%' is arithmetically
        correct and completely meaningless to a reader, so print the figures instead."""
        if not then:
            return f"{then:,.0f} to {now:,.0f}"
        if (now < 0) != (then < 0):          # sign flip
            return f"{then:,.0f} to {now:,.0f}"
        change = (now - then) / abs(then) * 100
        if abs(change) > 300:                 # tiny base -> runaway percentage
            return f"{then:,.0f} to {now:,.0f}"
        return f"{change:+.0f}%"

    return Paragraph(
        f"Year on year ({prior.year} to {rec.year}): sales {move(rec.sales, prior.sales)}, "
        f"operating profit {move(rec.operating_profit, prior.operating_profit)}, "
        f"borrowings {move(rec.borrowings, prior.borrowings)}, "
        f"reserves {move(rec.reserves, prior.reserves)}. Figures in Rs crore.", st["small"])


# ------------------------------------------------------------------ public API
def build_report(
    rec: ScreenerFinancials,
    combined: CombinedRisk,
    fin: FinancialScore,
    digital: Optional[DigitalPulse],
    narrative: str,
    recommendations: dict[Audience, list[Recommendation]],
    sector: str = "",
    prior: Optional[ScreenerFinancials] = None,
) -> bytes:
    """Render the two-page memo and return the PDF bytes."""
    st = _styles()
    generated = datetime.now().strftime("%d %b %Y %H:%M")
    buf = BytesIO()

    doc = BaseDocTemplate(
        buf, pagesize=A4, leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_BANNER_H + 8 * mm, bottomMargin=18 * mm,
        title=f"Foresight AI - {rec.company}", author="Foresight AI Analytics Engine",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(
        id="main", frames=[frame],
        onPage=lambda c, d: _draw_page(c, d, rec.company, generated))])

    story: list = []
    # --- page 1 ---------------------------------------------------------
    story.append(Paragraph(rec.company, st["h1"]))
    meta = f"{sector} &middot; " if sector else ""
    story.append(Paragraph(f"{meta}FY{rec.year} reported financials &middot; "
                           f"Analyst: Foresight AI Analytics Engine", st["sub"]))
    story.append(Spacer(1, 5 * mm))
    story.append(_score_block(combined, fin, st))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("KEY FINANCIAL METRICS", st["section"]))
    story.append(_metrics_table(rec, prior, st))
    story.append(Spacer(1, 2.5 * mm))
    story.append(_yoy_line(rec, prior, st))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("MARKET INTELLIGENCE SIGNALS", st["section"]))
    story.extend(_signals_block(digital, st))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("ANALYST SUMMARY", st["section"]))
    story.append(Paragraph(narrative, st["narr"]))

    # --- page 2 ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("WHY THIS SCORE - ALTMAN Z'' DECOMPOSITION", st["section"]))
    story.append(Spacer(1, 1.5 * mm))
    story.append(_decomposition(fin, st))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("The four terms are exact and sum to the Z'' score; green "
                           "reduces risk, red increases it.", st["small"]))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("RECOMMENDED ACTIONS", st["section"]))
    story.append(Spacer(1, 1.5 * mm))
    story.extend(_recs_block(recommendations, st))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "This report is generated by an AI analytics system and should be used as a "
        "supplementary tool alongside professional financial analysis. Financial scores "
        "are derived from reported financial statements using the Altman Z'' model. "
        "Market-intelligence signals may include illustrative data and are labelled where "
        "so; they are not a substitute for verified disclosure.", st["disc"]))

    doc.build(story)
    return buf.getvalue()

# Phase 2 Findings - Serving Indian Companies

The Phase 2 acceptance gate (the design notes) was: *can we score Indian non-financial
companies at all, and do the healthy controls land healthy?* Answering it changed the
serving architecture. This documents what we found and why.

---

## 1. The serving bridge works mechanically

`src/serving/screener.py` maps Screener.in's aggregated financials to the Polish `AttrN`
feature space. Screener's default statements are heavily aggregated (no inventory,
receivables, or current-assets line), so the current-ratio family is structurally NaN.
The bridge populates **46 of 63** serving features; the rest are genuine gaps the model
routes as NaN.

## 2. The GBM model does NOT transfer to Indian companies

This is the headline finding. Scored on real point-in-time data:

| Company | Truth | GBM percentile | Verdict |
|---|---|---|---|
| TCS FY19 | healthy | 2% | ok |
| Infosys FY19 | healthy | **47%** | too high |
| Jet Airways FY19 | **bankrupt** | **59%** | **unusable** |
| RCom FY18 | distress | 93% | ok |

Jet Airways FY19 - negative net worth (reserves −₹12,809 Cr), a ₹5,536 Cr loss - scored
at the **59th percentile**, barely above pristine Infosys at the 47th. The healthy/distress
ordering is too weak to rescale: max-healthy 47% vs min-distress 59% is a razor's edge,
and more companies would overlap. This is **not** display compression; it is domain shift.

### Why it fails (from SHAP on Jet)

Two mechanisms, both diagnosed directly:

1. **Missingness inverts across domains.** In Polish training, some features were
   *informative when missing* - distressed firms failed to report. At serving those same
   features are *structurally* absent for every Indian firm. The model reads the absence
   as the training signal, which for 3-Year Gross Profit and Receivables Turnover meant
   *healthy* - SHAP showed those NaNs contributing −1.9 and −1.3 **against** distress,
   cancelling the correct Interest-Coverage (+2.3) and Debt/Assets (+0.6) signals.
2. **Trees clamp, they don't extrapolate.** Jet's ratios are far outside the Polish range
   (Debt/Assets 2.14 vs a p95 of 0.92; negative equity). A tree's rightmost split caps
   the contribution - an out-of-range distress case looks like "a moderately bad Polish
   firm," not a catastrophe.

Median imputation did not fix it (Infosys got *worse*, jumping to the 72nd percentile);
retraining on the 46-feature serving set did not fix it either. The boundary itself does
not transfer.

## 3. Altman Z'' transfers cleanly - it becomes the serving anchor

Altman is **linear**, so out-of-range values **extrapolate** instead of clamping, and it
has no training distribution to carry across domains. On the same real companies:

| Company | Truth | Altman Z'' | Zone |
|---|---|---|---|
| TCS FY19 | healthy | +10.96 | Safe |
| Nestle FY19 | healthy (FMCG) | +3.22 | Safe |
| HUL FY19 | healthy (FMCG) | +4.13 | Safe |
| Jet Airways FY19 | bankrupt | **−17.25** | Distress |

Clean, wide separation. The irony worth stating out loud: the Altman module we built in
Phase 1 as *"the benchmark to beat"* is the robust serving engine. On its home turf
(Polish) the GBM beats it 0.71 to 0.15; across domains, the formula wins.

## 4. FMCG negative-working-capital gate - PASSED

The specific demo-fatal risk: Z'' weights WC/TA at **6.56**, and healthy FMCG companies
(Nestle, HUL) run *negative* working capital - so Altman could false-flag them, which is
as fatal as Jet scoring healthy. It does not: their EBIT/TA and RE/TA more than
compensate. Nestle at 3.22 is the tightest to the 2.60 grey boundary; a less-profitable
negative-WC name could dip into grey, so watch this if the roster changes.

## 5. Resulting architecture

- **Serving Financial Score** = Altman Z'', mapped to 0-100 with the zone thresholds as
  anchors. Robust, formula-based, cross-domain.
- **Serving explainability** = the exact 4-term Z'' decomposition (WC/TA, RE/TA, EBIT/TA,
  equity/liabilities). No approximation - the terms sum to the score.
- **Module 4/5 sliders drive Altman.** Linear ⇒ inherently monotonic; the Phase 1
  `monotone_constraints` machinery is retained as a Polish-domain showpiece.
- **The GBM + SHAP + calibration + Optuna work** is the Polish-domain proof that the
  method beats the benchmark, and the demonstration of explainability methodology.

### The answer

> "We built and validated a gradient-boosted distress model on the Polish corpus, where
> it beats the Altman benchmark 0.71 to 0.15. Applying it cross-border to Indian
> companies, we found the learned boundary under-flags out-of-distribution extremes -
> tree models clamp rather than extrapolate, and missingness means different things in
> the two domains. So for cross-border scoring we anchor on Altman Z'', which is robust
> by construction, and use it as the live engine. That is the sound way
> to deploy a model trained in one domain and applied in another."

## 6. Real point-in-time data curated so far

TCS, Infosys, Nestle, HUL, Asian Paints (healthy); Jet Airways FY18+FY19, RCom,
Future Retail (distress/watch). All from Screener.in, cross-checked against known
magnitudes. Full financial gate passes (§ above).

---

## 7. Module 2 - Digital Pulse (built)

Four signals, each on the same 0-100 risk scale and band language as the financial score,
so Module 3 fusion is a later weighted average. Every reading carries a **specific
explanatory datum**, not a restated metric.

### Signal priority (by evidence strength)

1. **Leadership Stability (anchor).** Dated public filings - a resignation happened on its
   date or it did not, so **zero selection-bias risk**. Weighted by seniority (auditor/CFO
   exits sharpest). Blueprint's "> 2 exits = red" preserved as the Elevated threshold.
2. **News Sentiment.** Fallback-first: Loughran-McDonald finance lexicon (no deps) as the
   default, FinBERT swapped in behind the same `SentimentScorer` interface. Leans on the
   30-day-vs-prior-30-day **trend**, not the level.
3. **Hiring** and **Employee Confidence** - soft, explicitly labelled illustrative
   (historical posting counts / review snapshots can't be reconstructed exactly).

### Selection bias = look-ahead in disguise

The one place rigor is hardest to hold: curating a distressed company's signals *knowing
it collapsed* manufactures a perfect signal. Guard: leadership events are dated filings
(no freedom); the headline rule is outcome-blind ("major-outlet coverage that month, take
what's there, don't filter by tone"); we lean on trend. An imperfectly-separating signal
is the result.

### Digital gate (fallback scorer) - PASSES

| Company | Digital composite | Band | Note |
|---|---|---|---|
| TCS | 6.4 | Healthy | all four signals green |
| Jet Airways (Mar 2019) | 76.7 | Critical | Goyal-family board exits (verified), hiring −85%, reviews falling |
| Future Retail (Feb 2020) | 36.6 | Watch | leverage + hiring stress; no verified board exits yet |

### Provenance (corrected after review)

An initial version overclaimed two things; both are now fixed, because "tests pass" is
not "survives the question *is this real, and how did you pick it?*":

1. **Future Retail is NOT a divergence showcase.** An earlier cut placed CFO/director
   resignations at Feb 2020 - but Future Retail's real board exodus was **2022**, during
   the Reliance-deal collapse. Those events were placeholders (`"CFO"`, `"board member"`),
   and they were what pushed FR's digital score to a fake "Elevated 53" diverging from its
   Watch financials. Removed. FR's digital score is **36.6 (Watch)** - matching its
   financial Watch (44). Distress building on both fronts, no manufactured divergence.
2. **Headlines are illustrative, not "outcome-blind collected."** They were authored
   knowing the outcomes; the docstring principle didn't make the process bias-free.
   Relabelled illustrative throughout; the leadership anchor now carries a `verified` flag
   and only the Goyal-family Jet exits (2019-03-25, well-documented) are marked verified.

**The real, well-sourced showcase is the Jet timeline** (Slide 4): the digital composite
climbs toward Critical across 2018 → Mar 2019, ahead of the April 2019 grounding, anchored
on the verified Goyal exit. Jet does not show a financial/digital *divergence* (both read
Critical) - its value is the **lead time** of the digital deterioration, which is the
version of the thesis.

**Outstanding:** verify the two unverified Jet independent-director
resignations against actual BSE filings (or drop them); the financial scores and the
Goyal-anchored leadership signal are already solid.

**Scope boundary honoured:** built the four gauges + digital composite. Module 3
(financial + digital fusion) is deliberately NOT built - the composite is fusion-ready and
stops there.

### FinBERT earns the torch dependency (validated)

Built fallback-first, then installed torch+transformers and swapped `ProsusAI/finbert` in
behind the same interface. On the curated headlines, FinBERT and the L-M lexicon **agree
on sign 13/13**, but FinBERT captures context the lexicon misses:

| Headline | L-M | FinBERT |
|---|---|---|
| "Jet Airways defers loan repayment, lenders in talks" | 0.00 (neutral) | **−0.35** |
| "Jet Airways misses loan repayment to consortium of banks" | 0.00 | **−0.11** |
| "Lessors move to deregister Jet Airways aircraft over dues" | 0.00 | **−0.14** |
| "Naresh Goyal steps down as Jet Airways chairman" | 0.00 | **−0.26** |

The lexicon scores **5 of Jet's 10 headlines neutral** - they contain no lexicon words -
while FinBERT reads all as negative from context. For a distress case the lexicon
*under-reads* the signal. So FinBERT is the primary scorer (`default_scorer()` prefers it),
the lexicon the never-breaks fallback.

Robustness note: on the aggregate 30-day Jet reading the two agree (composite 81.3 vs 81.9,
both Critical), so the demo conclusion holds under either scorer - the fallback is safe to
ship on, and FinBERT adds precision on top.

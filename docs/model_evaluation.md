# Model Evaluation

**In plain English:** this document is the proof behind the headline claim - that Foresight
finds failing companies far more sharply than the textbook Altman screen. It walks through
how the model was tested on ~10,000 real companies with known outcomes, why we grade it the
way we do (accuracy is a trap when failures are rare), how it compares to Altman at the same
catch rate, and every design choice we made against a measurement rather than a habit. If you
just want the business takeaways, the README has them; this is the receipts.

Measured on the Polish Companies Bankruptcy dataset (UCI id 365). All figures are
out-of-fold from stratified 5-fold CV with fold-internal imputation and resampling.
Reproduce with `notebooks/` or `src.models.train.cross_validate`.

---

## 1. Why accuracy is banned (the number to quote)

| Horizon | Rows | Distress | Base rate | Majority-class accuracy |
|---|---|---|---|---|
| 1year | 7,027 | 271 | 3.86% | **96.14%** |
| 5year | 5,910 | 410 | 6.94% | 93.06% |

A model that predicts "never distress" scores **96.14% accuracy** and is worthless.
Random-model reference points: PR-AUC = base rate (0.0386), ROC-AUC = 0.50.

## 2. Headline model performance

**THE number** - LightGBM, `serving` (63 features), 1-year horizon, shipping config
(native NaN, class weighting, no SMOTE, **slider-scoped monotonic constraints**),
**stratified 5-fold CV**:

| Metric | Value | vs random |
|---|---|---|
| **PR-AUC** | **0.780 ± 0.028** | 20x base rate (0.039) |
| ROC-AUC | 0.951 ± 0.020 | - |

The unconstrained model reaches 0.803, but constraints are non-optional - see §6d. The
0.023 difference is the price of sliders that cannot embarrass us on stage, and the
constrained model is also *more stable* across folds (±0.028 vs ±0.043).

PR-AUC is the headline. At a 3.9% base rate ROC-AUC flatters every model, so it is
reported but does not decide.

> **Quote 0.803, not 0.844.** A single 20% calibrated holdout (n=1,406) gives 0.844, but
> every other figure in this document is 5-fold CV with a standard deviation. Reporting
> the better single-split number as the headline would not survive "was that
> cross-validated?" - so it is labelled as what it is in §6c and never led with.

## 3. We beat the Altman incumbent by 11x on precision

The classical Altman Z'' threshold rule vs our model. Recall matches almost exactly
(0.5203 both), which makes this a genuinely like-for-like comparison:

| Approach | Precision | Recall | Flagged | True cases found |
|---|---|---|---|---|
| Altman Z'' < 1.10 | 0.089 | 0.5203 | 1,586 | 141 |
| **Foresight AI** | **0.979** | 0.5166 | **143** | 140 |

**Same detection, 11.0x better precision.** In the terms a credit team feels:
*the textbook rule makes you investigate 1,586 companies to find 141 real ones. Ours
finds 140 by investigating 143.*

The Z signal itself is real - median Z'' is 3.54 healthy vs 0.96 distress. It is the
**fixed threshold**, not the ratio, that fails.

## 4. Operating points (choosing a credit policy cutoff)

A credit team consumes a review queue, not a probability. Out-of-fold, final config:

| Target recall | Threshold | Precision | Flagged | % of book |
|---|---|---|---|---|
| 50% | 0.926 | 0.979 | 143 | 2.0% |
| 60% | 0.786 | 0.953 | 171 | 2.4% |
| 70% | 0.291 | 0.702 | 272 | 3.9% |
| 80% | 0.083 | 0.438 | 496 | 7.1% |
| 90% | 0.011 | 0.179 | 1,361 | 19.4% |

Even at 90% recall the queue is 19.4% of the book. This table is the input to the
portfolio view and **must be regenerated whenever the model config changes** -
it has already gone stale twice (imputation switch, then constraints).

## 5. Train/serve parity costs 0.04 PR-AUC

| Feature set | Features | PR-AUC |
|---|---|---|
| `all` | 64 | 0.653 |
| `serving` | 63 | 0.610 |

The only excluded attribute is `Attr55` (**absolute** working capital). It is
predictive but currency-denominated, so it cannot transfer PLN -> INR. We accept a
~0.04 PR-AUC cost to keep a model that can actually score Indian companies.

*Answer:* "We deliberately gave up four points of PR-AUC by dropping
absolute working capital, because it is a currency quantity that does not transfer
across economies. We preferred a model that can score the companies we ship."

## 6. SMOTE ablation - why the obvious move was dropped

"SMOTE **plus** class weighting" is the obvious way to handle imbalance. We tested it
properly as a 2x2 rather than a single on/off toggle, because a naive toggle confounds
two corrections: SMOTE rebalances the fold to 50/50 *and* `scale_pos_weight` upweights
positives ~25x, so "both on" is a ~25x over-correction rather than a fair test.

**Full 2x2, PR-AUC, `serving` feature set:**

| Model | Horizon | neither | SMOTE only | weight only | both |
|---|---|---|---|---|---|
| xgboost | 1year | **0.769** | 0.627 | 0.762 | 0.600 |
| xgboost | 3year | **0.650** | 0.516 | 0.633 | 0.461 |
| xgboost | 5year | **0.787** | 0.652 | 0.775 | 0.612 |
| lightgbm | 1year | **0.789** | 0.659 | 0.772 | 0.609 |
| lightgbm | 3year | **0.657** | 0.546 | 0.640 | 0.494 |
| lightgbm | 5year | 0.787 | 0.669 | **0.792** | 0.629 |

*Methodology note: the 2x2 was run with median imputation active in every cell (SMOTE
cannot consume NaN, so a fair comparison required it throughout). Imputation was dropped
afterwards on separate evidence - §6b. The comparison is therefore internally valid but
its absolute values sit ~0.03 below the final config; the `weight only` cell here (0.772)
corresponds to 0.803 once imputation is removed.*

**Conclusions, consistent across all 6 independent tests:**

1. **SMOTE is harmful on its own**, not merely via double-correction: -0.12 to -0.14
   PR-AUC even with `scale_pos_weight=1`. This is the load-bearing finding, and it
   rules out the "just don't stack them" explanation.
2. **Class weighting is ~neutral on PR-AUC** (-0.017 at 1year, well inside a 0.05 fold
   std; slightly positive at 5year). It shifts the operating point, not the ranking.
3. **Stacking both is worst in every cell** - the two corrections compound.

**Why SMOTE fails here:** it interpolates synthetic minority points in a 63-dimensional
space with extreme outliers and up to 39% missingness. Synthetic "companies" are
generated between real distressed firms in a space where that midpoint is not a
plausible balance sheet, so the boundary blurs. Gradient boosters also already handle
skew natively, so SMOTE adds distortion without adding information.

**Answer (stronger than the scripted one):**
> "We tested imbalance handling as a full 2x2 rather than assuming. SMOTE cost us
> 12-14 points of PR-AUC across two model families and three horizons, even without
> class weighting - synthetic interpolation in a 63-dimensional ratio space with 39%
> missingness generates balance sheets that don't exist. We evaluate on
> Precision-Recall because at a 3.9% base rate accuracy is meaningless, and we let the
> ablation pick the method rather than the convention."

**Status: awaiting user decision** - this contradicts a stated non-negotiable.
`use_smote` remains a parameter, so either choice is one argument away.

## 6a. Calibration (deferred to the fusion step, flagged now)

`scale_pos_weight` inflates predicted probabilities, and SMOTE does too. Modules 3/4/5
display a 0-100 score and run what-if and stress scenarios **on that score**, so it must
mean something, not just rank correctly. Before the score becomes load-bearing, wrap the
final estimator in `CalibratedClassifierCV` (isotonic or sigmoid) and derive the
displayed score from calibrated probabilities. Brier score is already tracked per fold
for exactly this. It is not a blocker.

Note this interacts with the §6 decision: since class weighting buys ~nothing on PR-AUC
but does cost calibration, "neither, then calibrate and tune the threshold" may be the
cleanest final architecture.

## 6b. Missingness is informative - do not impute

Distressed companies fail to report. `Attr27` (interest coverage) is absent for 311
firms, and its *absence* carries signal. Letting the boosters route NaN natively beat
median imputation in **all six** tested cells:

| Model | Horizon | median-impute | native NaN | Delta |
|---|---|---|---|---|
| xgboost | 1year | 0.7621 | 0.7858 | +0.0236 |
| xgboost | 3year | 0.6330 | 0.6604 | +0.0273 |
| xgboost | 5year | 0.7750 | 0.8044 | +0.0293 |
| lightgbm | 1year | 0.7723 | 0.8031 | +0.0307 |
| lightgbm | 3year | 0.6397 | 0.6541 | +0.0144 |
| lightgbm | 5year | 0.7919 | 0.8097 | +0.0178 |

Imputation also **inverted the what-if calculator**: with the median filled in, setting
interest coverage to a *worse* value *lowered* the displayed risk score, because the
imputed median was more benign than "missing".

## 6c. Calibration: sigmoid, not isotonic

| Method | Brier (raw → cal) | Score range | Saturation |
|---|---|---|---|
| isotonic | 0.01286 → 0.01216 | 0.0 - 100.0 | **855 at exactly 0, 79 at exactly 100** |
| **sigmoid** | 0.01279 → 0.01251 | **1.29 - 95.56** | none |

Isotonic wins marginally on Brier but saturates, and a gauge reading exactly 100 reads
as a bug in a live demo. Sigmoid holds ECE at 0.015 with slightly better PR-AUC (0.844).

Final headline: **PR-AUC 0.844** (calibrated sigmoid, LightGBM, `serving`, 1year holdout).

## 6d. Monotonicity - RESOLVED

**The problem was far worse than the first symptom suggested.** Measuring properly
(sweeping each metric across a grid for 40 companies and counting wrong-direction steps)
showed the unconstrained model violated economic direction constantly:

| Metric | Violations, unconstrained | After fix |
|---|---|---|
| Interest Coverage Ratio | **82.9%** | 0.0% |
| Current Ratio | 66.0% | 0.0% |
| Operating Return on Assets | 44.4% | 0.0% |
| Long-Term Debt to Equity | 42.0% | 0.0% |
| Quick Ratio | 34.5% | 0.0% |
| Equity Ratio | 25.6% | 0.0% |
| Return on Assets | 16.0% | 0.0% |
| Asset Coverage of Debt | 14.4% | 1.9% |
| Net Profit Margin | 12.5% | 0.0% |
| Debt to Assets | 10.5% | 0.0% |

Improving interest coverage *raised* the risk score on 83% of steps. Anyone moving
that slider would very likely have seen risk go the wrong way.

**Fix: `monotone_constraints`, scoped to user-facing metrics.** Scope matters - this is
a UI guarantee, not a global modelling goal:

| Scope | Constrained | PR-AUC (CV) | Cost |
|---|---|---|---|
| none | 0 | 0.803 | - |
| **slider (shipped)** | **10** | **0.780** | **-0.023** |
| all | 45 | 0.702 | -0.101 |

Constraining all 45 directional attributes costs 4x more for **no additional demo
safety**, because only exposed controls can be moved. Ambiguous ratios (turnover, days,
size, growth) are deliberately left free - a wrong constraint encodes false economics
and cannot be spotted by looking at the score.

**Residual, measured rather than assumed.** Three metrics show a nonzero violation
*rate*, so the precise claim is "≈0", not "0". What matters is magnitude:

| Metric | Violation rate | Max wrong move (score points) |
|---|---|---|
| Asset Coverage of Debt | 1.25% | 0.0000486 |
| Long-Term Debt to Equity | 0.50% | 0.0000059 |
| Return on Assets | 0.42% | 0.0000008 |
| *other seven* | 0.00% | 0 |

The largest wrong-direction move anywhere is **5e-05 score points** - roughly a
millionth of the gauge, far below one pixel. The constrained ensemble is monotone by
construction (a sum of monotone boosters composed with a monotone sigmoid), so this is
float accumulation across the 5 folds and the calibration transform, not a modelling
failure.

Accordingly the regression test asserts on **magnitude, not violation rate**
(`MonotonicityCheck.is_visible`, threshold 0.01 score points). A rate-based threshold
tuned just past an observed 1.25% would fail on a seed change while telling us nothing
about what a user sees.

Verified end-to-end - interest coverage 0.2 → 10.0 now moves the score 46.5 → 43.4
monotonically, and debt/assets 0.1 → 1.2 moves it 43.7 → 46.2.

Calibration is unaffected: ECE 0.0142 constrained vs 0.0149 unconstrained.

**Guarded by `test_every_slider_responds_in_the_correct_direction`, parametrised over
the full `SLIDER_FEATURES` set**, plus `explain.audit_sliders()` for a pre-demo check of
all ten at once. Any metric added to a what-if or stress control must be added to
`polish_schema.SLIDER_FEATURES` - a slider absent from that tuple is unguarded.

*Note for the stress test:* score movements are modest (~3 points across a full sweep)
because a single ratio rarely dominates. A visible move ("81 to 91 under rate stress")
needs the macro scenario to shift **several** correlated inputs at once, not one
slider. The stress model is built accordingly.

## 6e. Optuna tuning - 50 trials, a null result

**Protocol.** A stratified 20% test split is carved off first and Optuna never sees it.
The search optimises 5-fold CV PR-AUC on the training portion; improvement is measured
on the untouched test set. Tuning against the folds you then report is a known
optimistic bias -- the search partly memorises the split.

`scale_pos_weight`, `monotone_constraints`, and `monotone_constraints_method` are
**excluded from the search space** and enforced by `PROTECTED_PARAMS`. They encode
decisions already made against evidence; a search maximising PR-AUC alone would happily
discard the monotonic constraints for +0.02 and silently destroy the demo-safety
guarantee.

**Result - tuning did not meaningfully help.**

| Measurement | Baseline | Tuned | Delta |
|---|---|---|---|
| Held-out test split (n=1,406) | 0.8093 | 0.8244 | +0.0150 |
| **5-fold CV (full data)** | **0.7798** | **0.7813** | **+0.0015** |

The held-out number looks like a win; the cross-validated number shows it is not. A
+0.0015 gain sits well inside the ±0.025 fold standard deviation. **The reading
is that the hand-set hyperparameters were already near-optimal** and 50 trials of TPE
search could not beat them by more than noise.

Tuned params are marginally more *stable* (±0.0246 vs ±0.0282), which is the only
reason to prefer them.

**The valuable output was not better hyperparameters - it was a latent bug.**
Parameter importance came back dominated by one term:

| Parameter | Importance |
|---|---|
| `subsample` | **0.7915** |
| learning_rate | 0.0884 |
| min_child_samples | 0.0684 |
| *(all others)* | < 0.02 |

The optimum was `subsample ≈ 1.0` - *don't bag at all*. Investigating why exposed that
**LightGBM silently ignores `subsample` unless `subsample_freq >= 1`**, and the hand-set
baseline set `subsample=0.8` with no frequency. It was never bagging:

| Config | PR-AUC |
|---|---|
| baseline: `subsample=0.8`, no freq (**inactive**) | 0.7798 |
| `subsample=1.0` + `freq=1` (no bagging) | 0.7798 |
| `subsample=0.8` + `freq=1` (**bagging active**) | 0.7642 |
| `subsample=0.6` + `freq=1` (heavy bagging) | 0.7404 |

The first two rows are identical, which proves the no-op. Critically, **"fixing" the
dead parameter would have cost 0.016 PR-AUC.** Bagging genuinely hurts here: at a 3.9%
base rate, row subsampling drops distressed companies out of individual trees.

*Answer:* "We ran 50 Optuna trials on a training split with the test set held
out. Tuning bought us 0.0015 PR-AUC, inside fold noise - our hand-set parameters were
already near-optimal. What the search did surface was that our subsample parameter was
a silent no-op, and that activating it would have cost us 1.6 points, because bagging
discards minority-class rows at a 3.9% base rate."

Monotonicity is preserved under the tuned parameters (all ten sliders invisible, max
wrong move 6.8e-04 score points).

Artefacts: `models/optuna_lightgbm.json`, `models/optuna_lightgbm_trials.csv`.

## 7. Data quality notes

- `Attr37` ((current assets - inventories) / long-term liabilities) is **39.0% missing** --
  the single worst column. Needs an explicit strategy, not silent imputation.
- `Attr21` (sales growth YoY) is 23.1% missing; `Attr27` (interest coverage) 4.4%.
- 26 rows in 1year lack an Altman component and correctly yield NaN rather than a
  partial Z.
- Missingness is handled by median imputation **inside** the CV fold. Fitting the
  imputer before the split would leak validation distribution into training.

## 8. Leakage controls in place

1. SMOTE and imputation run inside the training fold only (imblearn `Pipeline`).
2. Validation folds retain the true 3.9% distress rate -- never resampled.
3. `scale_pos_weight` is computed from the training fold, not the full dataset.

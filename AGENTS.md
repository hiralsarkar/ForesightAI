# AGENTS.md - Foresight AI

**Read this file fully before writing any code in this repo.** It is the source of truth for scope, standards, and decisions - a distillation of the original competition blueprint, which lives in the originating conversation rather than in-repo. Everything load-bearing from it has been carried across here.

Where this file and the blueprint disagree, **this file wins**, and the disagreement must be recorded in the Decision Log below with its evidence. Two such amendments already exist (SMOTE, imputation) - both made against measurement and user-approved. Do not silently reverse them.

---

## 1. The Governing Question

Every module, chart, and sentence in this platform must help answer one question:

> **Should you change your decision today because this company's risk profile is changing?**

Not "will this company go bankrupt" - that is academic. The real question is whether a bank should tighten collateral, whether an investor should cut exposure, whether a supplier should demand advance payment instead of 90-day terms.

**Test for every feature:** does it help answer that question? If not, cut it.

**The thesis line:**
> "Financial statements describe what has already happened. Digital signals reveal what is starting to happen. Foresight AI reads both."

---

## 2. Product Standard

Target feeling: **Bloomberg Terminal meets McKinsey risk memo.** Software a credit risk team at HDFC Bank would open in a meeting. Not a data science dashboard. Not a notebook with sliders. A product.

**Palette (do not deviate):**

| Token | Hex | Use |
|---|---|---|
| `--bg-base` | `#0A1628` | Dark navy base |
| `--accent` | `#F59E0B` | Amber accent |
| `--text-primary` | `#FFFFFF` | Primary text |
| `--text-secondary` | `#94A3B8` | Secondary/label text |
| `--status-good` | green | Healthy / stable |
| `--status-watch` | amber | Watch |
| `--status-bad` | red | Elevated / critical |

**Three rules that apply everywhere in the UI:**

1. **Every number carries context.** "Current Ratio: 0.9" is worthless. "Current Ratio: 0.9 - industry median 1.4; less short-term liquidity than 78% of sector peers" is a finding. Every metric card = value + trend arrow (coloured) + one line of context.
2. **Every chart has a headline insight, not a title.** Not "Debt to Equity Over Time" but "Leverage has increased 2.4x over 24 months - well above the sector threshold." **Write the headline before building the chart.** The chart exists to prove the headline.
3. **The AI narrative must never read like a chatbot.** It reads like a senior credit analyst who just read the whole dashboard. Test on 10 companies; if any output reads like a filled-in template, the prompt is wrong.

---

## 3. Architecture Rules

- **`src/` modules, notebooks import them.** Notebooks are for demonstration and evaluation, never the implementation. A judge opening one 800-line notebook costs real points on technical depth.
- **No module writes to the global Python env.** Everything runs in `.venv`. Stack is pinned in `requirements.txt`.
- **Every scoring function is pure and testable.** Score computation must be importable and unit-testable without Streamlit.
- **Never let the demo crash.** Every external call (LLM, scrape, API) has a deterministic fallback path. Handle: company not found, CSV missing columns, API timeout, empty signal data.

```
src/
  features/    ratio computation, trend engineering, Altman Z
  models/      training, tuning, evaluation, SHAP
  signals/     news sentiment, hiring, leadership, reviews
  scoring/     combined risk score fusion
  narrative/   LLM narrative + rule-based fallback + recommendations
  reporting/   PDF executive report
app/           Streamlit application + custom CSS
notebooks/     evaluation & demonstration only
tests/         unit tests for scoring, features, fallbacks
data/raw|processed|demo
docs/          blueprint, model card, decision records
```

---

## 4. Data Strategy - Two Separate Problems

Keep these separate at all times.

### 4a. Training data - Polish Companies Bankruptcy (UCI id 365)

~10k companies, 5 forecasting horizons (`1year.arff`..`5year.arff`), binary bankruptcy label.

**CRITICAL - verified fact:** this dataset contains **pre-computed, anonymised financial ratios `Attr1`-`Attr64`**, NOT raw financial statements. The blueprint's "compute 25 ratios from the input data" does **not** apply to the training path - the ratios *are* the features. The ratio engine exists for the **Indian/serving path**, where we start from raw Screener.in financials.

Reframe the label as **"financial distress"** in all docs and presentation - more accurate to what the ratios capture.

**Three-class labelling:** derive `healthy / watch / distress` using an Altman-Z-style score in addition to the binary label. See §5 for the threshold trap.

### 4b. Demo data - current Indian companies (**non-financial only**)

**★ ROSTER REPLACED 2026-07-21.** The previous roster used *historical* cases (Jet Airways
2019, Reliance Comm 2018, Future Retail 2020). That forced illustrative data for hiring
and employee sentiment, because those cannot be reconstructed for a past date - which in
turn produced coverage gaps and disclosure badges in the demo. **Current companies do not
have that problem**: headcount is in quarterly results, employee ratings are live, and
news is dense. So the roster is now six *current* companies with **complete real data on
all four signals**. No gaps, no caveats, no badges.

Financials from Screener.in, FY2026 (year ended March 2026). Non-financial companies only
(a bank's balance sheet is out-of-distribution for an industrial ratio model - real credit
teams use CAMELS, and Altman excluded financial firms).

| Company | Sector | Combined | Fin / Dig | Role in the demo |
|---|---|---|---|---|
| SpiceJet | Airline | 86 Critical | 99 / 65 | Distress on every axis; **auditor resigned** (hard event) |
| Ola Electric | EV | 83 Critical | 90 / 74 | C-suite exodus + 5% layoff; positive equity, burning cash |
| Vodafone Idea | Telecom | 68 Elevated | 87 / 41 | +₹34,552cr headline profit hiding −₹1,44,101cr reserves |
| Vedanta | Metals | 36 Watch | 48 / 16 | Levered, but **signals improving** (4 ratings upgrades) |
| TCS | IT | 18 Healthy | 1 / 43 | Pristine books; workforce signal amber (−23.5k headcount) |
| Paytm | Fintech | 11 Healthy | 10 / 13 | **Recovery** - loss → profit, hiring again |

**Why these six:** they span three bands, show *three different shapes of distress*, one
positive divergence (Vedanta), one negative divergence within a healthy name (TCS), and
one recovery (Paytm). A roster where everything simply agrees would not demonstrate that
the platform discriminates.

**Assessment date is 2026-05-31 for all names** so signals and FY2026 financials describe
the same moment. Note this is load-bearing: the leadership window is 12 months, and moving
the date materially changes which real events fall in scope.

**Validation gate** (`demo_companies.validate_roster`, enforced by
`tests/test_serving.py::test_acceptance_gate`):
- SpiceJet, Ola Electric, Vodafone Idea must score **Elevated or Critical**
- Vedanta must score **Watch/Elevated**
- TCS and Paytm must score **Healthy** (and under 15, per `test_healthy_controls_are_clearly_healthy`)
- Every company must carry **all four signals** (`test_every_roster_company_has_complete_signal_coverage`)

**Signal sources:**

| Signal | Source | Feature |
|---|---|---|
| News sentiment | GDELT (free) or NewsAPI (100 req/day) | 90d headlines → FinBERT → 30d rolling avg + trend vs prior 30d |
| Leadership change | BSE India announcement portal (public, legally scrapeable) | C-suite/KMP exits last 6 months; red if > 2 |
| Hiring | Naukri active job count over time | Trend matters more than absolute → Growing/Stable/Contracting |
| Employee confidence | Public Kaggle historical Glassdoor dataset | Rating trend → Improving/Stable/Declining |

**On Glassdoor, be upfront:** "This module uses a publicly available historical dataset. A production version would connect to a licensed data feed." Mature and honest; judges respect it.

BSE is the uniquely India-specific source no competing team will think of. Prioritise it.

---

## 5. Known Traps (each one is a judge's probe)

**Trap 1 - Train/serve feature parity.** The model trains on Polish `Attr1-64`; the product scores Indian companies from Screener raw financials. If we train on all 64 anonymised attributes we get a model that **cannot score the demo companies at all** - different feature space. **Decide the shared feature set before training:** the intersection of (documented Polish attributes) ∩ (computable from Screener financials). The UCI attribute→economic-definition mapping is needed for this *and* for the plain-English SHAP labels. Get it before training, not after.

**Trap 2 - Wrong Altman Z variant/thresholds.** The blueprint says thresholds 1.81 / 2.99. Those belong to the **original Z**, which requires *market* value of equity - not available for these effectively-private Polish firms. If we compute the private-firm **Z''** variant (book equity, drops sales/TA), its cutoffs are **~1.1 / 2.6**, not 1.81/2.99. Applying original-Z thresholds to a Z'' score is exactly the kind of thing a data-science judge catches. Record which variant is used, per dataset, in the Decision Log.

**Trap 3 - Point-in-time integrity.** No future data used to explain a past prediction. Never 2020 data for a 2019 prediction. Prepared answer: *"All features used in each prediction are strictly point-in-time. We reconstructed the data pipeline to only use information available as of the prediction date."*

**Trap 4 - Overclaiming lead time.** Do **not** say "predicts distress 9 months in advance" unless point-in-time analysis rigorously demonstrates it. Say: *"our analysis of historical signals suggests earlier detection compared to financial statements alone."* A specific number you cannot back is a liability.

**Trap 5 - Dependency stack.** Global env has numpy 2.5 / pandas 3.0, which fight `shap`, `xgboost`, and `imbalanced-learn`. Pin a known-good stack in `.venv`. Cheaper now than debugging a SHAP import error after the pipeline exists.

---

## 6. Modules

| # | Module | Core requirement |
|---|---|---|
| 1 | **Financial Health Engine** | 25 ratios in 5 groups (Liquidity, Solvency, Profitability, Efficiency, Cash Flow) + **trend features** (1y and 2y change per key ratio) + explicit Altman Z shown alongside the ML model |
| 2 | **Digital Pulse Panel** | 4 gauges: News Sentiment, Hiring Activity, Leadership Stability, Employee Confidence. Label it **"Market Intelligence Signals"** in the UI (what a bank calls it) |
| 3 | **Combined Risk Score** | Top of dashboard. Semicircular gauge 0-100 + label (Healthy/Watch/Elevated/Critical). Financial 60% / Digital 40%, tunable |
| 4 | **SHAP Explainability** | TreeExplainer, waterfall per prediction, **business-language labels** + 3-sentence rule-based summary + what-if slider |
| 5 | **Macro Stress Testing** | 5 sliders: Interest Rate, GDP Growth, Inflation, Sector Credit Spread, USD/INR. **Live update, < 2s** |
| 6 | **AI Financial Narrative** | One paragraph, four sentences, credit-analyst voice |
| 7 | **Risk Mitigation Recommendations** | **Rule-based**, 15-20 rules, split *For Lenders/Investors* vs *For Management* |
| 8 | **Portfolio Risk Monitor** | 10-company sortable table + heatmap view, colour-coded rows |
| 9 | **Executive Report** | One button → 2-page PDF via ReportLab |

**Module detail that is easy to get wrong:**

- **M1 trend features are where the model wins.** D/E of 2.5 is concerning; D/E that went 0.8 → 1.6 → 2.5 is alarming. Direction and velocity carry as much signal as level.
- **M2 gauges need a specific explanatory datum below each**, not a restated metric. Not "Sentiment: Negative" but "Sentiment declining for 11 weeks. 68% of recent coverage mentions 'liquidity concerns' or 'debt restructuring'."
- **M3's story is usually the gap between components.** Financial 72 / Digital 31 = problems not yet visible in financials. Make that explicit in the UI: *"Digital signals are significantly weaker than financial ratios suggest. This divergence historically precedes financial deterioration by one to three quarters."*
- **M4 labels:** `x14` → "Interest Coverage Ratio", `x3` → "Working Capital / Total Assets". Cryptic names on a judge-visible chart break the professional illusion instantly. The 3-sentence summary is **rule-based, not LLM** - it must be precise and reliable.
- **M6 prompt discipline:** pass company name, industry, combined score + label, top-3 SHAP features *with values and direction*, sentiment score + trend, hiring trend, leadership flags, Altman Z. **Never ask the LLM to calculate anything** - pass all numbers as inputs. This is what prevents hallucinated financials in a live demo. Fallback to rule-based summary on API failure.
- **M7 rules are tied to actual values.** e.g. interest coverage < 1.5 AND D/E > 2.5 → *"Current debt servicing capacity is insufficient relative to leverage. Priority action: assess refinancing options or covenant renegotiation with existing lenders."*
- **M9 disclaimer is maturity, not weakness:** *"This report is generated by an AI analytics system and should be used as a supplementary tool alongside professional financial analysis."*

---

## 7. Non-Negotiables

1. **Class imbalance:** SMOTE **plus** class weighting, evaluated on **AUC-ROC and Precision-Recall - never accuracy**. Stratified 5-fold CV. Prepared answer: *"SMOTE for oversampling the minority class in training, plus class weighting in the model, and we evaluated using Precision-Recall curve rather than accuracy because accuracy is misleading on imbalanced datasets."*
2. **Optuna tuning**, ≥50 trials, results logged. Shows rigor.
3. **Point-in-time data integrity.** (Trap 3)
4. **SHAP labels in plain English.** No variable names on any judge-visible chart.
5. **Indian case studies validated before the day.** *(Updated 2026-07-20 - roster is non-financial only.)* **Jet Airways must score Elevated/Critical; TCS, Infosys, Asian Paints must score Healthy.** Plus the structural-missingness gate in §4b. Test the full roster.
6. **The dashboard must not crash.** Graceful errors on every edge case.
7. **Demo script memorised.** Not read, not improvised.

---

## 8. Anti-Goals

- ❌ Default Streamlit styling. Budget 4-5h on custom CSS - it converts directly to judge perception.
- ❌ 10 half-working modules. **7 polished and reliable beats 10 with 3 that break under pressure.**
- ❌ LLM narrative that reads generic across different companies. Outputs must differ in *substance*, not just company name.
- ❌ Monolithic notebook. Modular code + clear README signals practitioner, not student.
- ❌ Specific unbacked lead-time claims. (Trap 4)

---

## 9. Build Order

| Phase | Days | Deliverable |
|---|---|---|
| **1** | 1-4 | Polish data loaded & cleaned. Feature engineering (incl. trends). Model trained + evaluated. SHAP working. **← current phase** |
| 2 | 5-7 | Screener.in data for demo companies. FinBERT sentiment pipeline. BSE leadership scraper |
| 3 | 8-10 | Job posting signals. Combined score function. Streamlit scaffold + CSS. Search working end-to-end |
| 4 | 11-13 | All panels connected. Stress sliders live. LLM narrative + fallback |
| 5 | 14-15 | Portfolio view. PDF report. Yes Bank & Jet Airways timeline charts |
| 6 | 16-17 | Fix everything broken. Edge cases. **Rehearse demo ≥5 times** |

**Phase 1 scope guard:** training data only. **No Screener/BSE/Naukri/GDELT scraping in phase 1.** Writing a scraper this phase = drift.

---

## 10. The Indian Case Studies (the winning move)

**Yes Bank is the headline.** In 2018 the ratios still looked acceptable on the surface. But BSE filings show founder + multiple senior departures through 2018-19; sentiment turned sharply negative through 2019; hiring contracted. Combined signal should reach **Elevated Risk by mid-2019** - ahead of the March 2020 RBI moratorium. **The timeline chart showing this gap is the single most powerful visual in the presentation.**

- **Jet Airways** - different pattern: steeper, faster financial deterioration, negative working capital visible in the numbers, digital signals compounding through 2018→2019.
- **IL&FS** - financials looked structured but opaque; alternative signals (auditor concerns in filings, governance coverage) lit up before rating agencies.
- **TCS / Asian Paints** - must stay firmly green across the same periods. The contrast validates the model.

Build with **point-in-time data only.**

---

## 11. Demo Flow (6-7 min, memorised)

**Roster: six current companies, FY2026 financials, complete data on all four market
signals for every one of them.** No gaps, no caveats. They are chosen so the signals point
in genuinely different directions.

| # | Company | Combined | Fin / Dig | What it uniquely shows |
|---|---|---|---|---|
| 1 | SpiceJet | **86** Critical | 99 / 65 | Distress on every axis - auditor walked out |
| 2 | Ola Electric | **83** Critical | 90 / 74 | C-suite exodus + layoffs; young company burning cash |
| 3 | Vodafone Idea | **68** Elevated | 87 / 41 | Headline profit hiding negative net worth |
| 4 | Vedanta | **36** Watch | 48 / 16 | Levered financials, but signals **improving** |
| 5 | TCS | **18** Healthy | 1 / 43 | Pristine books, but the workforce signal bites |
| 6 | Paytm | **11** Healthy | 10 / 13 | Recovery - loss to profit, hiring again |

---

**0. Open on the Portfolio Monitor.** *"Six companies, scored this morning on FY2026
filings and live market signals. Red at the top, green at the bottom. A credit officer
starts here, not by looking up one name."*

**1. SpiceJet - everything screaming.** Combined **86**. Negative net worth, operating
loss. Then the signals: *"the statutory auditor resigned, the engineering workforce is
down 8%, and the coverage is unanimous - salary delays, a six-month furlough, emergency
funding talks."* Point at the auditor exit: *"That one is a hard event. Auditors leaving
is the sharpest governance tell there is, and our model floors the score on it rather
than letting softer signals average it away."*

**2. Ola Electric - a different distress.** Combined **83**, but note the shape: *"Equity
is still positive. This isn't an insolvent balance sheet, it's a young company burning
cash."* Signals: **CFO resigned 19 January**, CTO and CMO gone before him, **5% of staff
cut on 31 January**, registrations below 10,000 for a third month while a rival books
20,786. *"Three C-suite exits inside four months is the pattern that precedes disclosure
events."*

**3. Vodafone Idea - the number that lies.** *"FY2026 net profit: positive 34,552
crore."* Pause. *"And the model still says Elevated. Why?"* Show the decomposition:
retained earnings at **-144,101 crore**, equity negative. *"The profit is a one-off. The
balance sheet is destroyed. A ratio model reading the whole balance sheet is not fooled
by a single line of other income."*

**4. Vedanta - signals ahead of the financials, in the good direction.** Financials say
**Watch (48)** - genuinely levered. But digital reads **16, Healthy**: *"Four ratings
upgrades in three months - Fitch, S&P, CRISIL, ICRA - a completed demerger, and a top-100
workplace listing."* *"This is the divergence people forget: signals can lead **upward**.
The accounts describe a levered company. The market is telling you it is being fixed."*

**5. TCS - healthy, but not silent.** Combined **18**. *"Best balance sheet on the
board - Z of 6.9, interest cover in the hundreds."* Then: *"And yet the workforce signal
is amber. Headcount fell 23,500 in FY26, attrition rose to 13.7%, and there's a
1,388-crore restructuring charge."* *"The platform doesn't panic - TCS stays Healthy - but
it does not hide the one thing that moved."*

**6. Paytm - the recovery.** *"Everything so far has been decline. This is the other
direction."* Swung from a 663-crore loss to a **552-crore profit**, and is **hiring
again** after two years of cuts. *"Risk monitoring isn't only about catching falls. It's
about noticing when something has genuinely turned."*

**Close:** *"Six companies. Three risk bands. Distress that looks nothing alike, a levered
company being upgraded, a healthy one with a workforce problem, and a genuine recovery.
One question answered each time: should you change your decision today?"*

### Demo discipline

- **Stress-test Vedanta** (moves ~20 points) - the app defaults to it for that reason.
  SpiceJet is near-saturated; its sliders barely move.
- The **Case Study tab** is Ola Electric: its monthly signal trail against a single annual
  filing. The point is *frequency*, not a lead-time claim.
- **If asked why no banks:** *"Corporate ratio models don't apply to banks - they need
  CAMELS-style analysis, and Altman himself excluded financial firms. We'd rather score
  nothing than score it wrong."*
- **Great question to invite:** why FinBERT reads "NCLAT stays insolvency admission" as
  positive. News tone and risk direction are not the same thing - which is why verified
  hard events floor the score instead of being averaged against tone.

---

## 12. Slides (6 max)

1. **Problem** - "Indian banks wrote off over 10 lakh crore rupees in bad loans in the
   last decade." + "What if the warning signs were visible months earlier?" Let it breathe.
2. **Insight** - two columns: *Financial Statements* (what has already happened; annual;
   backward-looking) -> arrow -> *Market Signals* (what is starting to happen; continuous;
   forward-looking). "Foresight AI reads both."
3. **Product** - the **Portfolio Monitor** screenshot as the hero: six companies,
   red-to-green, every column populated. Insets: risk gauge, Digital Pulse, stress sliders.
4. **Six companies, six stories** - the table from SS11. This is the slide that shows the
   platform discriminates rather than just flagging everything.
5. **Two things a ratio alone misses** - side by side:
   *Vodafone Idea*: "+34,552 crore profit. Still Elevated." (one-off income vs a destroyed
   balance sheet) and *Vedanta*: "Levered on paper. Four upgrades in three months."
   (signals leading upward). Caption: *"Direction matters as much as level."*
6. **Market** - 2x3 grid: Banks, Investors, PE/VC, Corporates, Auditors, CFOs. Close:
   *"One platform. Six use cases. One question: should you change your decision today?"*

### What NOT to put on a slide

- Any specific **lead-time claim** ("flagged N months early") - not backed by our data.
- **Banks or private companies** - out of scope by design, and we have the answer ready.

---

## 13. Decision Log

Record every non-obvious choice here with its reason. Future sessions must not silently reverse these.

| Date | Decision | Reason |
|---|---|---|
| 2026-07-19 | Isolated `.venv`, stack pinned in `requirements.txt` (numpy 1.26.4 / pandas 2.2.3 / shap 0.46 / xgboost 2.1.3 / lightgbm 4.5) | numpy 2.5 / pandas 3.0 in global env are incompatible with shap/xgboost/imbalanced-learn (Trap 5). **Verified all import cleanly.** |
| 2026-07-19 | Polish data confirmed as pre-computed `Attr1-64` ratios, not raw financials | `.arff` headers are bare `@attribute AttrN numeric` with no definitions; economic definitions taken from UCI docs into `src/features/polish_schema.py`. Ratio engine therefore serves the *Indian* path only (Trap 1) |
| 2026-07-19 | **Altman variant = Z'' (private, sector-neutral), thresholds 1.10 / 2.60 - NOT 1.81 / 2.99** | Polish firms are unlisted; `Attr8` is *book* equity/total liabilities. Original-Z cutoffs require *market* equity and would mis-band every company (Trap 2). Z' also implemented for comparison. Components map exactly: X1=Attr3, X2=Attr6, X3=Attr7, X4=Attr8, X5=Attr9 |
| 2026-07-19 | Serving-parity feature set exposed via `serving_feature_set()`: 63/64 usable (26 direct, 37 derived) | Trap 1 guard. `Attr55` (absolute working capital) excluded as scale-dependent - does not transfer PLN→INR. Derived attrs are recoverable from Screener days-ratios (receivables = Debtor Days x Sales/365) |
| 2026-07-19 | Rows missing any Altman component yield NaN, never a partial Z | A Z built from 3 of 4 terms is not a Z; zero-filling would push firms into distress for a *data* reason, not a financial one (26 rows affected in 1year) |
| 2026-07-19 | **Non-negotiable #1 amended by measurement: class weighting only, SMOTE OFF.** User-approved. | Full 2x2 ablation (2 model families x 3 horizons): SMOTE costs 12-14 PR-AUC points *even with class weighting disabled*, so it is not a double-correction artefact. `use_smote` stays a parameter so the ablation is reproducible. See `docs/phase1_findings.md` §6 |
| 2026-07-19 | **No imputation - boosters route NaN natively.** | Missingness is informative: distressed firms fail to report. Native NaN beat median imputation in **all 6** tested cells (+0.014 to +0.031 PR-AUC). Imputation also *inverted* the what-if calculator, making worse interest coverage read as lower risk. SMOTE force-enables imputation since it cannot consume NaN |
| 2026-07-19 | **Calibration = sigmoid, not isotonic.** User-approved ("option 1 plus recalibrate") | Isotonic saturates: 855 companies scored exactly 0 and 79 exactly 100 - a gauge reading exactly 100 reads as a bug in a live demo. Sigmoid spans 1.29-95.56 with ECE 0.015 and slightly better PR-AUC. Score is a straight linear map of calibrated probability so "81" means 81% |
| 2026-07-19 | Narrative leads with *supports* for companies scoring < 25 | Describing a healthy company by its largest risk driver read as self-contradiction ("healthy… largest contributors are deteriorating X") |
| 2026-07-19 | **Monotonic constraints, scoped to the 10 user-facing metrics** (`SLIDER_FEATURES`) | Unconstrained model moved the wrong way on 82.9% of interest-coverage steps. Slider scope costs 0.023 PR-AUC; constraining all 45 directional attrs costs 0.101 for no extra demo safety. Ambiguous ratios left free - a wrong constraint encodes false economics invisibly |
| 2026-07-19 | Headline metric restated as **0.780 ± 0.028** (shipping config, 5-fold CV) | Was 0.803 unconstrained. Constraints are non-optional, so the shipped number is the honest one. Constrained model is also more fold-stable (±0.028 vs ±0.043) |
| 2026-07-19 | **Optuna: 50 trials run and logged (non-negotiable #2 satisfied). Result is a null.** | +0.0150 on a held-out split but only **+0.0015 on 5-fold CV** - inside the ±0.025 fold std. Hand-set params were already near-optimal. Do not claim tuning improved the model |
| 2026-07-19 | **`scale_pos_weight` / `monotone_constraints` excluded from the search space** (`tune.PROTECTED_PARAMS`) | A search maximising PR-AUC alone would discard the monotonic constraints for ~+0.02 and silently destroy the demo-safety guarantee. Evidenced decisions are not re-litigated by random search |
| 2026-07-19 | **LightGBM `subsample` was a silent no-op; left effectively disabled** | LightGBM ignores `subsample` unless `subsample_freq >= 1`. Baseline set 0.8 with no freq. *Activating* it costs 0.016 PR-AUC (0.7798 → 0.7642); heavy bagging costs 0.039. At a 3.9% base rate, row subsampling drops distressed rows out of trees. **Do not "fix" this parameter** |
| 2026-07-20 | **Demo roster is non-financial companies only. Banks/NBFCs removed, incl. headline Yes Bank.** User decision. | The industrial ratio model cannot score banks (no inventory/COGS/current ratio; asset turnover ~0.08 vs ~1.0 = out-of-distribution garbage). User wants every featured company scored end-to-end and cleanly. Non-bank distress cases are plentiful (Jet Airways, RCom, Future Retail, Videocon, Suzlon). Honest scope boundary a bank-risk judge respects - real banks use CAMELS, and Altman excluded financials from the Z-score |
| 2026-07-20 | **Headline distress case: Yes Bank → Jet Airways.** ✅ §11 and §12 rewritten 2026-07-20 around a **five-company arc** (TCS → Nestle → Future Retail → RCom → Jet), every figure verified from the live app | Jet Airways is an airline (industrial structure, scores end-to-end), collapsed April 2019, universally recognisable, and carries both a financial-engine story and the digital-signal story the Yes Bank case was chosen for |
| 2026-07-20 | Point-in-time financials **hand-curated into versioned CSVs**, not live-scraped | WebFetch's summariser editorialises on numbers (invented a "forward-looking projections" gloss on the Yes Bank fetch); Screener's current columns may be restated vs as-originally-reported. Curation is more reliable *and* more point-in-time-honest (Trap 3) |
| 2026-07-20 | **★ ARCHITECTURE: serving Financial Score is anchored on Altman Z'', NOT the ML model. The GBM model does not transfer cross-domain.** | Measured on real data: bankrupt Jet Airways FY19 (negative net worth) scored at the **59th percentile** by the GBM - unusable - while healthy Infosys sat at the 46th. Root cause (from SHAP): features that were *informative-when-missing* in Polish (3-yr GP, receivables turnover) map to "healthy" at serving and cancel the true distress signals; and trees *clamp* Jet's out-of-range ratios to "moderately bad Polish firm". Altman is **linear**, so extremes **extrapolate** instead of clamping, and it has no training distribution to transfer - it separates cleanly: TCS +10.96 / Nestle +3.22 / HUL +4.13 (Safe) vs Jet −17.25 (Distress). This is domain shift, handled the graduate-level way: anchor serving on a formula robust by construction. See `docs/phase2_findings.md` |
| 2026-07-20 | **FMCG negative-WC gate PASSED** - Altman does not false-flag Nestle/HUL | Z'' weights WC/TA at 6.56; Nestle (−20 WC days) and HUL (−32) run negative working capital while pristine. Their EBIT/TA and RE/TA more than compensate: both land Safe. Nestle at 3.22 is the tightest (grey starts 2.60) - watch if a less-profitable negative-WC name joins the roster |
| 2026-07-20 | **ML model kept as: (a) Polish-domain showpiece (beats Altman 0.71 vs 0.15), (b) SHAP methodology.** Not the serving score. | Honest framing: "we detected a Polish-trained model under-flags cross-border extremes, so we anchor serving on a formula robust by construction." Do NOT write this as "we couldn't make the ML transfer" - same facts, wrong voice |
| 2026-07-20 | **Module 4/5 sliders drive the Altman score, not the GBM** | Coherence requirement: the gauge shows Altman, so the sliders must move Altman. Altman is linear → inherently monotonic (Phase 1 `monotone_constraints`/`SLIDER_FEATURES` become Polish-showpiece-only), and its 4-term decomposition is an **exact** waterfall - a better serving explanation than GBM SHAP on a number we don't display. Avoid the incoherent build: gauge=Altman, slider=GBM |
| 2026-07-20 | **Module 2 built fallback-first: Loughran-McDonald lexicon default, FinBERT swapped in behind a `SentimentScorer` interface** | Don't let a heavy torch install block the pipeline. L-M is the finance-standard lexicon (VADER mis-scores "debt/liability/aggressive"), so even the fallback is defensible. Wire + test everything on the fallback, then add FinBERT as an enhancement. Same "demo never breaks" discipline as the Phase 1 narrative fallback |
| 2026-07-20 | **Signal priority by defensibility: Leadership (anchor) > Sentiment > Hiring/Employee (soft, illustrative)** | Leadership = dated public filings, zero selection-bias risk. Hiring/Glassdoor historical data can't be reconstructed exactly → labelled illustrative in the readings. Don't over-invest in the soft two |
| 2026-07-20 | **Selection bias treated as look-ahead; headlines relabelled ILLUSTRATIVE (not "outcome-blind collected")** | A rigor review caught the overclaim: headlines were authored knowing outcomes, so the process is not bias-free - a docstring rule doesn't change that. Lean on trend; treat level as illustrative. Leadership events carry a `verified` flag; only the Goyal-family Jet exits are verified public filings |
| 2026-07-20 | **Future Retail is NOT a divergence showcase - corrected. Its digital = Watch (37), matching financial Watch (44).** | The original "financials Watch, digital Elevated 53" divergence was manufactured by **placeholder/anachronistic** FR leadership events (real FR board exodus was 2022, not the Feb-2020 assessment date). Removed. Do NOT reintroduce fabricated exits to recreate the divergence. Present FR honestly as "watchlist, distress building on both fronts" |
| 2026-07-20 | **Real digital showcase = the Jet timeline (lead time), not a divergence** | Jet reads Critical on both financial and digital, so there's no divergence. Its value is that the digital composite climbs toward Critical through 2018→Mar 2019 ahead of the April 2019 grounding - the honest form of "digital reveals what's starting to happen." Anchored on the verified Goyal exit |
| 2026-07-20 | **Module 3 fusion: renormalize over available components; NO manufactured divergence** | Only 3/8 companies have digital signals. Digital=0 with fixed 60/40 would drag financial-only companies toward healthy (same missingness bias as Phase 2). So combined = financial when digital absent, UI states it. All 3 dual-signal companies *agree* (no cross-sectional divergence exists) - narrative fires "divergence" only on genuine band disagreement, computed never assumed |
| 2026-07-20 | **Dashboard: Streamlit, slider path routed through Altman (linear), GBM never on the slider** | Streamlit reruns the whole script per interaction. `@st.cache_resource` for model+FinBERT, `@st.cache_data` for scores. Stress sliders recompute the linear Altman formula (instant, <2s trivially met). Verified: all 6 spine panels render with correct data (M3 gauge, M1 cards, M2 pulse, M4 Altman waterfall, M5 stress) |
| 2026-07-20 | **Streamlit launch needs `--server.fileWatcherType none`** | The watcher introspects every `transformers` vision module → hundreds of torchvision ImportErrors + stalled first render. Disabling it fixes both. Config is in the *primary* working dir's `.claude/launch.json` (name: `foresight`), not ForesightAI's |
| 2026-07-20 | **M4 serving explainability = Altman 4-term waterfall, NOT GBM SHAP** | The waterfall shows the exact terms that drive the displayed Altman score (WC/TA, RE/TA, EBIT/TA, equity/liab), summing to Z''. GBM SHAP is the Polish-domain methodology showpiece, a separate concern from per-company serving explanation |
| 2026-07-20 | **Stress-test BUG FIXED: the leverage slider was dead** (caught in review) | `compute_features` derives `total_liabilities = total_assets - equity` and never reads `borrowings`, so shocking borrowings moved zero Altman terms. Modelled the leverage leg as a **releveraging** (debt up, reserves/equity down by the same amount, total assets constant) - worsens equity/TL and RE/TA, monotonic for every company, no working-capital artifact. Guarded by `tests/test_app.py::test_leverage_shock_leg_is_not_dead` |
| 2026-07-20 | **App default company = Future Retail, not Jet** | Jet is saturated at 100, so the stress panel shows CHANGE +0 on first load (looks broken). Future Retail is mid-range, leveraged (both sliders bite), and has digital signals (all panels populate) |
| 2026-07-20 | **Future Retail shows a MILD divergence under FinBERT (digital Elevated 52 vs financial Watch 44) - honest but scorer-dependent and on illustrative data** | With the L-M fallback FR's digital is 37 (agreement); FinBERT reads its distress headlines more negatively → 52 (one band worse). The narrative fires legitimately (computed, hedged "can precede… look closer"), but the digital leg rests on illustrative sentiment/hiring. Do NOT promote this to a validated divergence showcase - same lesson as the earlier fabricated-events correction. It illustrates the concept; it is not proof |
| 2026-07-20 | **★ The Portfolio Monitor (M8) is the deck's hero visual, NOT the Jet timeline** | The portfolio table rests on **verified** financials (real Screener data, Altman-computed) and gives the clean red-to-green contrast that reads as "a bank would use this". The Jet timeline is a *supporting* "anatomy of a collapse" panel |
| 2026-07-20 | **★ Jet timeline = CONVERGENCE, never a lead-time claim (Trap 4 guard)** | Measured: Jet's financial score is **flat at 100 across the whole 2018-19 window** (FY2018 already negative equity), while digital climbs 27→77 and only reaches Critical in **Mar 2019, one month before the grounding**. So financials led by ~10 months - Jet is NOT "digital leads financials". Worse, the pre-Mar-2019 digital line is entirely *illustrative* (hiring/reviews/authored headlines); the one **verified** signal (Goyal board exit, 25 Mar 2019) is **coincident** with the crisis. Any "N months of digital lead time" claim would be unbacked. Shipped framing: "every warning system was firing; no one acted until the planes stopped", with an explicit on-screen disclaimer that we make no lead-time claim and digital's value is *continuity between sparse annual reports*. Guarded by `test_jet_financials_were_already_critical_throughout` |
| 2026-07-20 | **No company in the dataset shows the blueprint's Slide-4 "gap" on verified data** | Jet has no gap (financials earliest); Future Retail's gap is FinBERT-dependent on illustrative signals. Decide the deck accordingly - lead with the portfolio table. Do not manufacture a gap |
| 2026-07-20 | **M6 narrative: LLM path passes only pre-computed numbers; deterministic fallback is the live path** | No `ANTHROPIC_API_KEY` configured, so rule-based is what ships today; the LLM path works if a key appears. System prompt forbids the model from calculating anything - that is the guard against hallucinated financials on stage. Fallback varies by band + dominant Altman term + signal state, so outputs differ in *substance* (guarded by `test_narrative_differs_in_substance_not_just_name`) |
| 2026-07-20 | **M7 = 22 rule-based recommendations, never LLM** | They sit beside a number a credit committee may act on, so they must be reproducible and unable to invent a fact. Split For Lenders/Investors vs For Management; every rule cites the company's actual values |
| 2026-07-20 | **★ M7 bug fixed: payables are NOT debt, and negative working capital is not always a problem** | `Attr2` (total liabilities/assets) includes trade payables, so leverage rules keyed on it told **debt-free HUL** its "leverage is material", and the working-capital rule told a healthy FMCG name to "close the working capital gap" - when supplier financing *is* the FMCG model. Leverage rules now key on `borrowings/total_assets`; the WC rule is gated on the band not being Healthy. Wrong advice to a healthy company is a credibility killer in the demo. Guarded by `test_debt_free_company_is_not_told_it_is_levered` and `test_healthy_fmcg_not_told_to_close_working_capital_gap` |
| 2026-07-21 | **M6 narrative now runs live via OpenRouter** (`openai/gpt-oss-20b:free`) | Key in gitignored `secrets.local.json`, loaded by `src/narrative/llm_config.py`. The `:free` slugs for llama/deepseek are deprecated; nemotron models leak chain-of-thought; gpt-oss-20b returns clean 4-sentence prose. Prompt still passes numbers-only; rule-based fallback intact. **Rotate key before public GitHub.** |
| 2026-07-21 | **Narratives pre-cached for OFFLINE demo safety** (`data/demo/narrative_cache.json`, committed) | `generate()` resolves cache -> live LLM -> rule-based. Cache key = SHA of the exact prompt (incl. sector), so a data change invalidates it. Pre-warm via `summary.prewarm_cache()` using the FinBERT path so keys match the app. Verified: no key + network raising still yields 6/6 AI narratives. Sectors moved to `demo_companies.SECTORS` (one source) so app + prewarm prompts match |
| 2026-07-21 | **Pitch deck built** (`deck/ForesightAI.pptx` + PDF, 6 slides) via pptxgenjs, product palette, visually QA'd through PowerPoint COM export | The blueprint's §12 spec, realised. Portfolio table is the hero (slide 3); slide 5 contrasts Vodafone Idea (profit but flagged) vs Vedanta (levered but upgraded) |
| 2026-07-21 | **Evaluation notebook** (`notebooks/01_model_evaluation.ipynb`) built + executed via `_build_evaluation.py` | Imports from `src/` only (architecture rule). Covers accuracy-trap, CV PR-AUC, Altman benchmark (10x precision), SMOTE ablation, the cross-domain pivot, live roster scores, SHAP. Regenerate by re-running the builder |
| 2026-07-20 | **Reliance Comm signals filled with VERIFIED public record** (NCLT/NCLAT orders); assessed as of 2018-06-15 | Real dated events: NCLT admitted Ericsson's insolvency plea 2018-05-15; IRP appointed and board powers suspended 2018-05-18; NCLAT stayed the admission 2018-05-30. Apr-2019 director exits deliberately **excluded** as future-dated. Frame as **confirmation, not prediction** - RCom's FY2018 financials already scored Critical *before* these dates. An NCLT admission is the distress event, not an early warning |
| 2026-07-20 | **Did NOT fabricate signals for the healthy four** (Nestle, HUL, Infosys, Asian Paints) | A neutral test search ("Nestle India news 2019") returned product launches and awards - nothing signal-worthy. Turning that into a risk score would be manufacturing data. Also: the leadership scorer defaults empty→"Stable", so adding an uncovered company would assert a verified-looking negative from absent data. **Uncovered signals are now omitted entirely, not scored as Stable** |
| 2026-07-20 | **"Signal Coverage" badge replaces the bare dash** - the actual answer to "gaps look bad" | `Full / Partial / Financials only` + provenance (`verified / part-verified / illustrative / no feed`). Disclosed coverage is what a real credit platform reports; an empty cell reads as broken. This is a presentation fix, not a data fix, and it is the honest answer wherever real data genuinely doesn't exist |
| 2026-07-20 | **★ Hard-event floor in the digital composite: facts are not averaged away by tone** | FinBERT scores *"NCLAT stays insolvency admission, allows management to function"* at **+0.69 (positive)** - tone-correct, risk-inverted. That dragged RCom's news signal to 25, averaged against a board-suspension reading of 96, and **de-escalated RCom from Critical to Elevated purely by adding data**. Now a verified `BOARD_SUSPENSION`/`AUDITOR_EXIT` sets `hard_event=True` and floors the composite at its own reading. General lesson worth stating to judges: **news tone ≠ risk direction** ("rescue plan agreed", "lenders restructure" are positive-tone, severe-distress). Guarded by `test_adding_verified_distress_data_never_de_escalates` |
| 2026-07-20 | **M9 report: prints the Altman decomposition, not a SHAP waterfall; no fabricated trajectory charts** | The blueprint's page 2 asks for SHAP + debt/profitability trend charts. Our serving explanation is the Altman 4-term decomposition (the GBM doesn't drive the displayed score), and we curated **one fiscal year for 7 of 8 companies** so multi-year trajectories don't exist - printed as a year-on-year text line where a prior year exists, and an explicit "single reported year available" note otherwise. Do not invent trend charts to fill the space |
| 2026-07-20 | **PDF glyph + percentage fixes** | `''` (U+2033) and `→` (U+2192) are absent from Helvetica's encoding and **silently drop**, rendering "Altman Z''" as "Altman Z ". Replaced with ASCII. Separately, Jet's operating profit going +24 → −3,660 printed as "−15350%" - arithmetically right, meaningless to a reader; sign flips and tiny bases now print absolute values. Guarded by `tests/test_report.py` |
| 2026-07-20 | Added a **Grey-zone/Watch lender rule** | Future Retail (Watch, 42% borrowings) originally produced **no lender guidance at all** - coverage 2.09x just missed the `<2.0` trigger and its zone is Grey not Distress. The watchlist band is precisely where a lender needs direction |
| 2026-07-20 | **Digital signals are a time-series (multi-date), not snapshot** | Latest reading → Module 2 gauges; trajectory → Jet case-study timeline (Slide 4, deck's key visual). `SignalSeries.as_of(when)` enforces point-in-time (Trap 3) |

### ⚠️ Trap 6 - trend features are not learnable from the training data

**Discovered in Phase 1.** Module 1 calls trend features "where the model wins" and the
build order lists "feature engineering (incl. trends)". This is **not buildable on the
Polish data**: the five horizon files carry no company identifier, so the same firm
cannot be tracked across years and 1y/2y change-per-ratio cannot be computed.

The consequence is the mirror of Trap 1: a model trained *without* trend features cannot
consume them at serving time either. So "the model wins on trends" cannot be literally
true of the ML component, and claiming it would be indefensible.

Two pseudo-trend ratios are already baked into the data (`Attr21` sales(n)/sales(n-1),
`Attr24` 3-year gross profit / total assets) and do carry velocity signal - `Attr21`
ranks 5th in global SHAP importance.

**Decision:** trends become a **serving-side display and narrative layer**, not an ML
input. Module 1 charts the trajectory (D/E 0.8 → 1.6 → 2.5), Module 6's narrative and
Module 7's rules reason over it, and the Yes Bank timeline is built from it - all of
which is where trends are most persuasive anyway. They must not be described as model
features.

*Defensible answer:* "Trend direction drives our narrative and recommendation layer.
The distress model itself is trained point-in-time on ratio levels, because the training
corpus is anonymised and cannot be tracked per company across years - so we don't claim
the model learns trajectories it never saw."

### ✅ Monotonicity - RESOLVED (2026-07-19)

The unconstrained model violated economic direction on **82.9%** of interest-coverage steps - improving coverage *raised* risk. Ten user-facing metrics all violated between 10% and 83%. Fixed with `monotone_constraints` scoped to `polish_schema.SLIDER_FEATURES`; violations now ~0 and the behaviour is guarded by `test_sliders_respond_in_the_correct_direction`.

**Rule for future work:** monotonicity is a **UI guarantee**, not a global modelling goal. Constraining all 45 directional attributes costs 0.101 PR-AUC; constraining only the 10 exposed ones costs 0.023 for identical demo safety.

> **Any metric added to a Module 4 what-if or Module 5 stress slider MUST be added to `SLIDER_FEATURES` first.** Otherwise that control can move the score the wrong way in front of judges.

Ambiguous ratios (turnover, days, size, sales growth) are deliberately left unconstrained. A wrong constraint encodes false economics and is invisible in the score. Notably `Attr21` sales growth is free *because* hypergrowth-then-collapse is itself a distress pattern - Byju's is in our own demo set as exactly that case.

### ✅ Module 5 macro stress-testing - BUILT (2026-07-21)

Three macro levers (interest rate, inflation, GDP) + two company-specific (operating profit, leverage), in a pure testable module `src/scoring/stress.py`. **Interest is below EBIT**, so a rate shock does not touch WC/TA or EBIT/TA - it raises interest expense (shown as the coverage change, the headline credit channel) and, sustained over a 3-yr horizon, erodes equity. Inflation/GDP flow to operating profit via **sector-specific elasticities** (an airline has no pricing power; a commodity producer passes inflation through; IT reprices). All shocks compose on the line items and recompute Altman once. **Dropped USD/INR** (helps exporters, hurts importers - a uniform slider moves IT the wrong way) **and sector credit spread** (duplicates the rate channel). Direction guarded: adverse macro raises risk for all 6 companies (`test_macro_adverse_scenario_raises_risk_for_every_company`).


### Verified dataset facts (measured, not assumed)

| Horizon | Rows | Distress | Rate | Majority-baseline accuracy |
|---|---|---|---|---|
| 1year | 7,027 | 271 | 3.86% | **96.14%** |
| 5year | 5,910 | 410 | 6.94% | 93.06% |

**The 96.14% is the argument for banning accuracy** - a model that never predicts distress scores 96%. Quote this number when asked about class imbalance.

**Altman Z'' as a standalone classifier (1year)** - this is the incumbent our model must beat:
`precision 0.089 | recall 0.520 | F1 0.152` - flags 1,586 companies to catch 141 of 271 true cases.
Median Z'' is 3.54 (healthy) vs 0.96 (distress): the *signal* is real, the *threshold rule* is unusable for a credit team. That gap is exactly what the ML adds, and why Module 1 shows both.

**Worst missingness (1year):** `Attr37` 39.0%, `Attr21` 23.1%, `Attr27` 4.4%. Attr37 needs an explicit strategy - do not silently impute.

---

## 14. What Wins

Technical accuracy gets top 10. Explainability pushes to top 5. **First place is whether judges walk out still thinking about it.**

That happens when the story is airtight end to end - when every element answers the same question, when the Yes Bank case study makes someone uncomfortable because they realise the signals were there and nobody was reading them, when the narrative sounds like a person wrote it, when the PDF looks worth paying for.

**You are not presenting a machine learning model. You are presenting a decision support system for people who manage billions of rupees of risk.** Build and present accordingly.

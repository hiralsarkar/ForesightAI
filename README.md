# Foresight AI

**Corporate Financial Health Intelligence** - early-warning distress detection that reads
financial statements *and* digital signals.

> Financial statements describe what has already happened.
> Digital signals reveal what is starting to happen. Foresight AI reads both.

The platform answers one question: **should you change your decision today because this
company's risk profile is changing?**

---

## The problem

Annual accounts are a lagging indicator. By the time a company's distress reaches its
filed financials, a lender, investor or supplier has usually already been exposed to it for
months. The classical screen for this - the Altman Z-score - is decades old and, at a
realistic base rate, floods a credit team with false positives.

Foresight AI catches distress earlier and cleaner: it scores the financials with a
calibrated model, reads the market signals that move *between* filings, and fuses the two
into a single explainable risk score.

## How it works

Two independent read-outs, fused into one decision.

- **Financial engine** - a gradient-boosted model produces a 0-100 distress score, anchored
  to the Altman Z'' formulation and decomposed term-by-term, so every score comes with the
  reason behind it rather than a black-box number.
- **Digital Pulse** - four market-intelligence signals on the same 0-100 scale: news
  sentiment (FinBERT over dated headlines), leadership stability (dated exchange filings),
  hiring trend, and employee confidence. Each reading carries a specific datum, not a
  restated metric.
- **Combined score** - a weighted fusion that renormalises over the signals actually
  available and shows the two legs separately, because the *gap between them* is the story:
  distress often shows in the market before it reaches the accounts.

Around that core the platform adds a portfolio surveillance view, a live macro/company
stress test, an AI-written analyst summary with a deterministic fallback, rule-based
recommended actions, and a one-click PDF executive report.

## Results

Trained on the Polish Companies Bankruptcy dataset (UCI id 365), 1-year horizon.

| Metric | Value |
|---|---|
| **PR-AUC (5-fold CV)** | **0.780 ± 0.028** |
| ROC-AUC (5-fold CV) | 0.951 ± 0.020 |
| Expected calibration error | 0.015 |
| Random baseline (PR-AUC = base rate) | 0.039 |

All figures are stratified 5-fold cross-validation on the shipping configuration.

**Against the classical benchmark**, at near-identical recall:

| Approach | Precision | Recall | Flagged | Found |
|---|---|---|---|---|
| Altman Z'' < 1.10 | 0.089 | 0.5203 | 1,586 | 141 |
| **Foresight AI** | **0.979** | 0.5166 | **143** | 140 |

Same detection, **11x better precision**. The textbook rule makes a credit team investigate
1,586 companies to find 141 real ones; this finds 140 by investigating 143.

## Why it transfers: two separate problems

The obvious question is *you train on Polish companies, then score Indian ones - why does
that hold?* Because these are two deliberately separated problems, and the model never
crosses the gap the question assumes.

**Training path (the statistical distress signal).** The model learns from the Polish
Companies Bankruptcy set: ~10k firms, five forecast horizons, a binary distress label.
Critically, this dataset ships **pre-computed, anonymised financial ratios** (`Attr1`-`Attr64`),
not raw statements from any one country. The model learns the *shape of distress in
dimensionless financial ratios* - leverage, coverage, working-capital adequacy,
profitability decay - a structural, currency- and geography-neutral signal. It is not
learning "what a Polish balance sheet looks like."

**Serving path (the Indian companies).** Indian names never touch the training ratios. We
start from raw Screener.in FY2026 financials, run them through our own ratio engine, and map
into a **serving-parity feature subset** - only the ratios that are both computable from
public Indian filings and present in the training schema. The transfer is therefore
ratio-to-ratio on a shared, dimensionless feature space, never statement-to-statement across
jurisdictions.

**Verified vs. modelled.** Financials are real (Screener.in, FY ending March 2026). The four
digital signals are drawn from real public reporting for the same moment - quarterly-result
headcount, published review-platform ratings, dated exchange leadership announcements, real
news coverage. The Altman anchor and its 1.10/2.60 cutoffs are the classical formulation for
unlisted firms. Nothing in the demo is synthetic.

## Design decisions that hold up

Each was made against measurement, not convention. Full rationale in `AGENTS.md`.

- **Accuracy is never reported.** A model predicting "never distress" scores 96.1%. PR-AUC
  is the headline; ROC-AUC is secondary at a 3.9% base rate.
- **SMOTE is off.** A 2x2 ablation showed it costs 12-14 PR-AUC points across two model
  families and three horizons, even without class weighting.
- **Nothing is imputed.** Missingness is informative - distressed firms fail to report.
  Native NaN routing beat median imputation in all six tested cells.
- **Monotonic constraints on the interactive features.** The score must move the right way
  when a user stresses a ratio; without constraints the model moved the *wrong* way on 82.9%
  of interest-coverage steps. Scoping constraints to the slider features brings that to ~0,
  at a cost of ~2 PR-AUC points (0.803 → 0.780) - non-negotiable for a tool people interact
  with.
- **Altman Z'' with 1.10/2.60 cutoffs**, not the original Z's 1.81/2.99 - these are unlisted
  firms with no market equity.
- **Sigmoid calibration**, so a displayed score of 81 means an 81% modelled probability.
- **Serving-parity feature set**, so the model can actually score real companies.

## Model selection

Hyperparameters are searched, not guessed - and the search **finds no configuration worth
adopting over the hand-set one**. A 50-trial Optuna study (TPE) optimises 5-fold CV PR-AUC
under a clean disjoint-test protocol: class weighting, NaN routing and the monotonic
constraints held fixed (`src/models/tune.py`, logged to `models/optuna_lightgbm.json` and
`optuna_lightgbm_trials.csv`). At tune time the winner looked ~0.015 PR-AUC ahead on a
single uncalibrated split.

**That edge does not survive the shipping pipeline.** Run the tuned config through the full
calibrated ensemble that actually ships and the two are statistically indistinguishable -
every gap is an order of magnitude below the ±0.028 fold spread:

| Metric (calibrated, end-to-end) | Hand-set | Optuna (2x size) |
|---|---|---|
| CV PR-AUC (headline) | 0.780 | 0.781 |
| Held-out PR-AUC | **0.828** | 0.822 |
| Held-out ROC-AUC | **0.975** | 0.968 |
| Calibration (ECE, Brier) | **better** | worse |

The reliable 5-fold estimate is level; on the held-out split the hand-set model is actually
ahead on both ranking metrics *and* better calibrated - at half the trees and leaves. So the
apparent tune-time gain was a protocol artifact (a single uncalibrated booster on less data),
not a real improvement. We ship the hand-set config deliberately, and the tuned params stay
logged for audit.

## The dashboard

```bash
.venv/Scripts/python.exe -m streamlit run app/main.py --server.fileWatcherType none
```

Then open http://localhost:8501. The `--server.fileWatcherType none` flag is required - the
watcher otherwise tries to introspect every `transformers` vision module. First load
downloads FinBERT (~440MB) once.

Three views:

- **Company Analysis** - the combined risk gauge, financial ratio cards with the Altman Z''
  anchor, the four Digital Pulse signals, the exact Altman term-by-term decomposition
  ("why this score"), an AI analyst summary, recommended actions, and a live macro + company
  stress test.
- **Portfolio Monitor** - surveillance table across the roster, highest risk first,
  colour-coded. Built on verified financials; the strongest visual.
- **Case Study (Ola Electric)** - the Digital Pulse through the year against the single
  annual financial data point: the C-suite exits, the January 2026 CFO resignation, the 5%
  workforce cut and the collapsing retail registrations were all observable *before* the
  filing that confirmed them.

The AI analyst summary runs live via OpenRouter with a deterministic rule-based fallback,
and is **pre-cached** to `data/demo/narrative_cache.json` so the demo produces real AI
narratives with no network (resolution order: cache → live LLM → rule-based). The API key
lives in a gitignored `secrets.local.json` - rotate or remove it before making the repo
public.

## Deliverables

| Item | Location |
|---|---|
| Dashboard | `streamlit run app/main.py --server.fileWatcherType none` |
| Pitch deck (6 slides, editable + PDF) | `deck/ForesightAI.pptx`, `deck/ForesightAI.pdf` |
| Evaluation notebook (executed, with plots) | `notebooks/01_model_evaluation.ipynb` |
| Executive report (per company, PDF) | Download button on the Company tab |
| Decision log & full rationale | `AGENTS.md`, `docs/` |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Download the dataset (UCI id 365) and extract the five `.arff` files into `data/raw/`, then
run the test suite:

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

### Score a company in code

```python
from src.features.load_polish import load_horizon
from src.models.calibrate import fit_calibrated, risk_score, band
from src.models import explain

df = load_horizon(1)
model, report, features = fit_calibrated(df)

X = df[features]
score = risk_score(model.predict_proba(X.to_numpy("float64"))[:, 1])

print(score[0], band(score[0]))
print(explain.summarise(model, X, features, row=0, score=score[0], company="Acme Ltd"))
print(explain.contributions_frame(model, X, features, row=0, top_n=5))
```

## Repository layout

```
src/features/    polish_schema.py   attribute -> business label + serving availability
                 load_polish.py     .arff loader, class balance, missingness
                 altman.py          Altman Z'' / Z', zones, three-class labels
src/models/      train.py           leak-safe CV, class weighting, operating points
                 calibrate.py       probability calibration, 0-100 score, bands
                 tune.py            Optuna search, disjoint-test protocol
                 explain.py         SHAP, business labels, narrative, what-if
src/signals/     the four Digital Pulse signals + composite
src/scoring/     combined score fusion, macro/company stress test
src/serving/     raw-financials -> serving-parity features, roster, screener
app/             Streamlit dashboard (main.py, theme.py, scoring_service.py)
tests/           136 guardrail tests
docs/            findings and rationale
```

## Honest limits

Stated plainly, because a score is only worth trusting if you know what it does and does not
claim.

- **Bands come from banding, not a 3-class classifier.** The model trains binary;
  healthy / watch / distress bands are derived by thresholding the calibrated probability. A
  deliberate choice, stated as one.
- **Trends are displayed, not modelled.** The anonymised training ratios carry no time
  dimension, so trend arrows are a serving/display-layer feature, not model inputs.
- **SHAP explains the full ensemble, and the one residual is disclosed.** The waterfall
  averages SHAP across all five calibrated folds, which sums exactly (to ~1e-15, in log-odds
  space) to the ensemble's mean margin - so it attributes the model behind the score, not a
  single fold. The only remaining gap to the displayed 0-100 number is the sigmoid
  calibration transform, which is monotonic and so preserves every direction and ranking the
  waterfall shows.

---

*This system is a supplementary analytics tool and should be used alongside professional
financial analysis, not in place of it.*

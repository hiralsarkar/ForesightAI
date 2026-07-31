# Foresight AI

**See corporate distress months before the filing does.**

Foresight AI reads a company's financial statements *and* its digital footprint - news,
leadership moves, hiring, employee sentiment - and turns them into one decision you can make
today instead of once a year: **is this company's risk changing right now?** Financial
statements tell you what already happened. Digital signals tell you what is starting to
happen. Foresight reads both.

[![Live demo](https://img.shields.io/badge/live%20demo-open%20app-FF4B4B?logo=streamlit&logoColor=white)](https://foresightai.streamlit.app)
[![Evaluation notebook](https://img.shields.io/badge/evaluation-notebook-F59E0B?logo=jupyter&logoColor=white)](notebooks/01_model_evaluation.ipynb)
[![Pitch deck](https://img.shields.io/badge/pitch-deck%20(6%20slides)-0A1628)](deck/ForesightAI.pdf)
[![Tests](https://img.shields.io/badge/tests-136%20passing-22C55E)](tests/)

**Try it live at [foresightai.streamlit.app](https://foresightai.streamlit.app)** - four
tabs, including a Review Economics calculator that tells you exactly how deep to review your
book.

---

## Why it matters

Today a credit team, lender, or supplier learns a company is in trouble the same way
everyone else does: when the filing lands, months too late. Their fallback screen, the
decades-old Altman Z-score, is so noisy that at a realistic 3.9% distress rate it turns
"find the distressed companies" into "investigate 1,586 to find 141." That review queue is
the real, recurring cost of the status quo.

Foresight replaces the queue with a ranked, explainable shortlist:

| At matched recall | Companies flagged | Real distress caught | Precision |
|---|---|---|---|
| Textbook Altman screen | 1,586 | 141 | 8.9% |
| **Foresight AI** | **143** | **140** | **97.9%** |

**Same catch. One-tenth the review queue. 11x the precision.**

And it puts a rupee value on the decision. Tell it what a missed default costs you and what
a review costs, and it computes the cost-optimal policy live:

> At Rs 50 lakh per missed default and Rs 1 lakh per review, review your top **18%** and
> catch **90%** of all distress for **Rs 26.6 Cr**, against the classical screen's best of
> **Rs 65.7 Cr**. A Rs 39 Cr swing, recomputed instantly for your own numbers in the
> **Review Economics** tab.

## What you get

A four-tab dashboard (live link above):

- **Company Analysis** - one combined risk gauge, the financial ratios behind it, four live
  market signals, and a plain-English "why this score" a credit committee can act on.
- **Portfolio Monitor** - your whole book ranked by risk, worst first.
- **Case Study** - watch the signals move month by month ahead of a real filing. For Ola
  Electric: the CFO resignation, the 5% layoff, and collapsing sales were all visible
  *before* the annual accounts confirmed them.
- **Review Economics** - the cost-optimal review policy, with sliders for your own cost of a
  miss and a review.

Plus a one-click PDF executive report per company, a live portfolio stress test, and an
AI-written analyst summary.

## How it works

**Two signals, one score, always explainable.**

- **Financial engine** - an Altman Z'' distress score, decomposed term by term so every
  point traces back to a real ratio: leverage, coverage, working capital, profitability.
- **Digital Pulse** - four independent market reads on the same 0-100 scale: news sentiment
  (FinBERT), leadership stability (exchange filings), hiring trend, employee sentiment. Each
  is a specific, sourced fact, not a restated financial metric.
- **Combined score** - fused with the two legs always shown side by side, because the *gap*
  between them is the story: distress usually shows in the market before it reaches the
  accounts.

**The right engine for each job.** We built a machine-learning distress model *and* the
linear Altman engine, then tested both on real, named bankruptcies. The ML model wins
decisively in controlled tests (the 11x above), but on live balance sheets with values
outside its training range, gradient-boosted trees clamp instead of extrapolating: it scored
bankrupt Jet Airways at the 59th percentile, one rank above pristine Infosys. Altman, being
linear, extrapolates cleanly and separated the same companies with no edge cases:

| Company | Reality | Altman Z'' |
|---|---|---|
| TCS | healthy | +10.96 |
| HUL | healthy | +4.13 |
| Jet Airways | bankrupt | **-17.25** |

So the product **serves with Altman**, and keeps the ML model as the benchmark that proves
the method beats the textbook screen 11x over. Shipping the engine that works on the data in
front of you, not the one with the prettier lab score, is the whole point.

## The proof

The 11x result is not a lucky split. The distress model is trained on the Polish Companies
Bankruptcy dataset (~10k firms), 1-year horizon, stratified 5-fold cross-validation. The
headline is PR-AUC, because at a 3.9% base rate accuracy is meaningless - predicting "never
distress" scores 96.1%.

| Metric | Value |
|---|---|
| **PR-AUC (5-fold CV)** | **0.780 ± 0.028** |
| ROC-AUC | 0.951 ± 0.020 |
| Calibration error (ECE) | 0.015 |
| Random baseline | 0.039 |

Every number here is reproduced end to end in the
[evaluation notebook](notebooks/01_model_evaluation.ipynb) - nothing is hand-typed.

## Built on evidence, not convention

Each call was made against a measurement:

- **No SMOTE** - a 2x2 ablation showed it costs 12-14 PR-AUC points.
- **Nothing imputed** - missingness is itself a distress signal; native NaN routing beat
  imputation in every test.
- **Monotonic constraints on every slider** - without them the score moved the *wrong* way
  on 82.9% of stress-test steps.
- **Calibrated** - a score of 81 means an 81% modelled probability, not just a rank.
- **Tuning that says no** - a 50-trial Optuna search looked ahead at tune time but was
  statistically indistinguishable once run through the shipping pipeline, so we kept the
  simpler model and logged the rest.

Full rationale, including the cross-domain test in detail, is in [`AGENTS.md`](AGENTS.md)
and [`docs/`](docs/).

## Run it locally

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app/main.py --server.fileWatcherType none
```

Open <http://localhost:8501>. The app runs on cached FinBERT scores
(`data/demo/sentiment_cache.json`), so it starts instantly with no model download.

## Deploy your own live demo

No GPU, no API key, no training data at runtime - `requirements.txt` excludes
torch/transformers because the app replays cached FinBERT scores.

1. [share.streamlit.io](https://share.streamlit.io) -> sign in with GitHub.
2. **New app** -> repo `hiralsarkar/ForesightAI`, branch `main`, file `app/main.py`.
3. Name it `foresightai` and **Deploy**.

## Deliverables

| Item | Location |
|---|---|
| Live dashboard | [foresightai.streamlit.app](https://foresightai.streamlit.app) |
| Pitch deck (6 slides) | `deck/ForesightAI.pdf` |
| Evaluation notebook (executed) | `notebooks/01_model_evaluation.ipynb` |
| Executive report (per company, PDF) | Download button on the Company tab |
| Decision log and full rationale | `AGENTS.md`, `docs/` |

## Repository layout

```
src/features/    Polish data loader, Altman Z'' engine, feature schema
src/models/      leak-safe CV, calibration, Optuna tuning, SHAP explainability
src/signals/     the four Digital Pulse signals + composite
src/scoring/     combined score fusion, macro/company stress test
src/serving/     raw financials -> Altman serving score, roster, review economics
app/             Streamlit dashboard (main.py, theme.py, scoring_service.py)
tests/           136 guardrail tests
docs/            findings and rationale, including the cross-domain test in full
```

## Scope

A decision-support tool built to sit alongside professional analysis, not replace it. Risk
bands come from a calibrated probability; trend arrows are a display layer, since the
training data carries no time dimension; SHAP explains the full 5-fold ensemble, one
monotonic step from the number you see.

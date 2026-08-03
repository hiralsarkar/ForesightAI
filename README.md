# Foresight AI

See corporate distress months before the filing does.

Foresight AI reads a company's financial statements and its digital footprint - news,
leadership moves, hiring, employee sentiment - and turns them into one decision you can make
today instead of once a year: is this company's risk changing right now? Financial
statements tell you what already happened. Digital signals tell you what is starting to
happen. Foresight reads both.

Live app: [foresightai.streamlit.app](https://foresightai.streamlit.app). The
[evaluation notebook](notebooks/01_model_evaluation.ipynb) reproduces every number below.

## Why it matters

A credit team, lender, or supplier usually learns a company is in trouble the same way
everyone else does: when the filing lands, months too late. The fallback screen, the
decades-old Altman Z-score, is noisy enough that at a realistic 3.9% distress rate it turns
"find the distressed companies" into "investigate 1,586 to find 141." That review queue is
the real, recurring cost of the status quo.

Foresight replaces the queue with a ranked, explainable shortlist:

| At matched recall | Companies flagged | Real distress caught | Precision |
|---|---|---|---|
| Textbook Altman screen | 1,586 | 141 | 8.9% |
| Foresight AI | 143 | 140 | 97.9% |

Same catch, one-tenth the review queue, 11x the precision.

It also puts a rupee value on the decision. Tell it what a missed default costs you and what
a review costs, and it computes the cost-optimal policy live: at Rs 50 lakh per missed
default and Rs 1 lakh per review, review your top 18% and catch 90% of all distress for
Rs 26.6 Cr, against the classical screen's best of Rs 65.7 Cr. That Rs 39 Cr swing is
recomputed instantly for your own numbers in the Review Economics tab.

## What you get

A four-tab dashboard:

- **Company Analysis** - one combined risk gauge, the financial ratios behind it, four live
  market signals, and a plain-English "why this score" a credit committee can act on.
- **Portfolio Monitor** - your whole book ranked by risk, worst first.
- **Case Study** - the signals moving month by month ahead of a real filing. For Ola
  Electric, the CFO resignation, the 5% layoff, and collapsing sales were all visible before
  the annual accounts confirmed them.
- **Review Economics** - the cost-optimal review policy, with sliders for your own cost of a
  miss and a review.

Plus a one-click PDF report per company, a live portfolio stress test, and an AI-written
analyst summary.

## How it works

Two signals, one score, always explainable.

- **Financial engine** - an Altman Z'' distress score, decomposed term by term so every
  point traces back to a real ratio: leverage, coverage, working capital, profitability.
- **Digital Pulse** - four independent market reads on the same 0-100 scale: news sentiment
  (FinBERT), leadership stability (exchange filings), hiring trend, employee sentiment. Each
  is a specific, sourced fact, not a restated financial metric.
- **Combined score** - fused with the two legs always shown side by side, because the gap
  between them is the story: distress usually shows in the market before it reaches the
  accounts.

### Why Altman serves, and the ML model benchmarks

This pairs a machine-learning distress model with the linear Altman engine, tested against
real, named bankruptcies. The ML model wins decisively in controlled tests (the 11x above).
But gradient-boosted trees clamp at the edge of their training range instead of
extrapolating, so on live Indian balance sheets - values outside anything in the Polish
training data - they lose separation:

| Company | Reality | ML model, percentile |
|---|---|---|
| TCS FY19 | healthy | 2% |
| Infosys FY19 | healthy | 47% |
| Jet Airways FY19 | bankrupt | 59% |
| RCom FY18 | distress | 93% |

Bankrupt Jet lands one rank above pristine Infosys, too weak to act on. Altman, being
linear, extrapolates instead of clamping and separates the same companies with no edge case:

| Company | Reality | Altman Z'' | Zone |
|---|---|---|---|
| TCS FY19 | healthy | +10.96 | Safe |
| HUL FY19 | healthy | +4.13 | Safe |
| Nestle FY19 | healthy | +3.22 | Safe |
| Jet Airways FY19 | bankrupt | -17.25 | Distress |

So the product serves with Altman and keeps the ML model as the benchmark that proves the
method beats the textbook screen 11x over. The exploration mapped exactly where a tree
model's assumptions break on out-of-distribution data, and shipped the one that holds on a
live balance sheet.

## What it scores today

The live dashboard scores six current Indian companies, worst risk first. Combined fuses the
Altman financial score with the four Digital Pulse signals; scores run 0-100, higher = higher
risk.

| Company | Sector | Financial | Digital | Combined | Band |
|---|---|---|---|---|---|
| SpiceJet | Airline | 99 | 65 | 86 | Critical |
| Ola Electric | Electric Vehicles | 90 | 74 | 83 | Critical |
| Vodafone Idea | Telecom | 87 | 41 | 68 | Elevated Risk |
| Vedanta | Metals & Mining | 48 | 16 | 36 | Watch |
| TCS | IT Services | 1 | 43 | 18 | Healthy |
| Paytm | Fintech | 10 | 13 | 11 | Healthy |

TCS shows the fusion earning its keep: pristine financials (1) but a digital leg at 43 from
workforce signals, the gap you only catch by reading both legs.

## The proof

The 11x result is not a lucky split. The distress model is trained on the Polish Companies
Bankruptcy dataset (~10k firms), 1-year horizon, stratified 5-fold cross-validation. The
headline is PR-AUC, because at a 3.9% base rate accuracy is meaningless: predicting "never
distress" scores 96.1%.

| Metric | Value |
|---|---|
| PR-AUC (5-fold CV) | 0.780 ± 0.028 |
| ROC-AUC | 0.951 ± 0.020 |
| Calibration error (ECE) | 0.015 |
| Random baseline | 0.039 |

Every number is reproduced end to end in the
[evaluation notebook](notebooks/01_model_evaluation.ipynb).

Each modelling call was made against a measurement, not a convention:

- **No SMOTE** - a 2x2 ablation showed it costs 12-14 PR-AUC points.
- **Nothing imputed** - missingness is itself a distress signal; native NaN routing beat
  imputation in every test.
- **Monotonic constraints on every slider** - without them the score moved the wrong way on
  82.9% of stress-test steps.
- **Calibrated** - a score of 81 means an 81% modelled probability, not just a rank.
- **Tuning that says no** - a 50-trial Optuna search was statistically indistinguishable from
  the hand-set model once run through the shipping pipeline, so the simpler model shipped.

Full rationale, including the cross-domain test in detail, is in [`docs/`](docs/).

# Foresight AI

**Live corporate risk scoring for any listed Indian company.** Pick a company and it pulls
the financials and the latest news off the internet in real time, scores the risk with the
classic Altman Z-Score fused with a live news-sentiment signal, explains every point, and
shows - on real bankruptcies - that it would have flagged the trouble early.

Live app: [foresightai.streamlit.app](https://foresightai.streamlit.app).

## What it does

- **Live Lookup** - pick any NSE company. It fetches the financials (Screener) and recent
  news (Google News) live, scores them into one risk read, and shows the Altman decomposition
  and the headlines it scored. Nothing is hard-coded - the whole pipeline runs on the spot.
- **Hindsight** - five real collapses replayed, each merging the Altman Z trajectory with the
  dated signals that led the accounts: auditor exits, rating actions, board changes, news shocks.
  RCom (distress flagged FY2016, three years early) and Suzlon (crisis then recovery) are Altman
  wins; DHFL and IL&FS are NBFCs where Altman is blind and only the signals led (IL&FS was rated
  AAA until the end); Manpasand is a clean auditor-resignation warning, a year early.
- **Company Analysis, Portfolio, Case Study, Review Economics** - deep-dive views on a curated
  set of six current companies.

## How it works

Two legs, one score, always explainable:

- **Financial** - the original 1968 Altman Z-Score, all five components, using the market value
  of equity for listed companies. A trusted, decades-old distress formula, computed live and
  decomposed term by term so every point traces to a real ratio.
- **Market signals** - a five-signal pulse for the tracked companies: credit rating, leadership
  changes, news sentiment, hiring and employee confidence. A rating cut to default or an auditor
  exit is treated as a hard event and floors the score. These often move before the accounts do.
- **Combined** - the two fused into one 0-100 risk read, with the legs shown side by side,
  because the gap between them is the story.

## The honest pitch

We do not claim to beat Altman - the financial engine *is* Altman. The value is making it
**live, automated, comprehensive and explainable for any company on demand**, adding market
signals that can lead the accounts, and proving on history that it catches real distress early.

We did test whether a trained model could beat the formula (see the notebooks): models trained
on foreign bankruptcy data (Polish, Taiwanese) do not transfer to Indian balance sheets, and a
model trained on scraped Indian companies matches the formula rather than beating it - which is
exactly why the trusted linear Altman score is the right anchor.

## Under the hood

```
src/foresight.py       the scoring engine (Altman Z, signals, fusion, stress)
src/screener_live.py   live financials + market cap from Screener
src/live_news.py       live news sentiment from Google News
app/main.py            the Streamlit dashboard
data/indian/           scraped Indian training set + the historical backtest
notebooks/             the model-building and evaluation story
```

Run locally: `streamlit run app/main.py`.

## Scope

A decision-support tool, not a substitute for professional analysis. It uses public data
(Screener, Google News) fetched live for the demo; a production version would use licensed feeds.

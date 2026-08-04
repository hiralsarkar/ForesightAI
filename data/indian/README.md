# Indian training data

A small, hand-built dataset of Indian companies for training the distress model on the
market it actually scores - which foreign datasets (Polish, Taiwanese) could not do.

## Files

- `nclt_cases.csv` - the label source: 832 insolvency petitions disposed by the NCLT/NCLAT
  under the Insolvency and Bankruptcy Code, hand-collected by the IGIDR Finance Research
  Group ([ifrogs.org](https://ifrogs.org/releases/nclt_data.html)). This identifies which
  companies went into distress.
- `companies.csv` - 21 listed companies (12 healthy, 9 distressed) with their latest-year
  financials scraped from [Screener.in](https://www.screener.in). The distressed set is
  drawn from known insolvency/default cases whose current financials still show the distress
  (negative equity or heavy losses); the healthy set spans IT, FMCG, autos, metals and
  infra, including levered-but-solvent firms so the model learns the middle ground.
- `train_india_model.py` - builds the four Altman ratios via the engine's own
  `compute_features`, trains a logistic regression, and tests it on the six demo companies.

## Why this exists

A model trained on the Polish bankruptcy data does not transfer to Indian balance sheets
(the trees clamp out-of-range values; a linear model can't anchor the healthy companies).
The Taiwanese data is pre-normalised to [0, 1], so it can't score raw Indian ratios at all.
Training on Indian companies fixes it, because the scale and accounting finally match.

## Result

| Training data | Demo companies scored correctly |
|---|---|
| Polish (logistic) | 4 / 6 |
| Taiwanese (logistic) | 3 / 6 |
| **Indian (this set)** | **6 / 6** |

The India-trained model places all six correctly, including the borderline Vedanta (Watch)
and the healthy TCS and Paytm (~9% risk).

## Honest limits

- Small (21 companies). The leave-one-out score is optimistic on a set this clean.
- It uses the latest financials, so it is a concurrent distress *detector*, like Altman,
  not a leading *predictor*. A predictive version would use pre-event financials.
- To grow it: add rows to `companies.csv` (same columns) and re-run the script. The label
  source in `nclt_cases.csv` lists hundreds more candidates.

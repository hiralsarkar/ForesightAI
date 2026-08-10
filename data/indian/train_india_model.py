"""Train the distress model on Indian companies and test it on the six demo companies.

Reads companies.csv (financials scraped from Screener.in, labelled healthy/distress),
turns each company into the engine's four Altman ratios, trains a logistic regression,
and checks it on the six demo companies in the app.

Run from the repo root:  .venv/Scripts/python.exe data/indian/train_india_model.py
Grow the dataset by adding rows to companies.csv and re-running.
"""
import csv, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score
import foresight as f

FEATS = ["Attr3", "Attr6", "Attr7", "Attr8"]   # the four Altman ratios (WC, RE, EBIT, equity)
HERE = pathlib.Path(__file__).resolve().parent


def company_features(row):
    """One CSV row -> the four Altman ratios, via the engine's own feature builder."""
    num = lambda k: float(row[k])
    fin = f.ScreenerFinancials(
        company=row["company"], year=int(row["year"]),
        sales=num("sales"), expenses=num("expenses"), operating_profit=num("operating_profit"),
        other_income=num("other_income"), interest=num("interest"), depreciation=num("depreciation"),
        profit_before_tax=num("profit_before_tax"), net_profit=num("net_profit"),
        equity_capital=num("equity_capital"), reserves=num("reserves"), borrowings=num("borrowings"),
        other_liabilities=num("other_liabilities"), total_assets=num("total_assets"),
        fixed_assets=num("fixed_assets"), working_capital_days=num("working_capital_days"))
    fe = f.compute_features(fin)
    return [fe.get(c, np.nan) for c in FEATS]


def main():
    rows = list(csv.DictReader(open(HERE / "companies.csv", encoding="utf-8")))
    X = np.array([company_features(r) for r in rows], "float64")
    y = np.array([int(r["label"]) for r in rows], "int8")
    print(f"India training set: {len(rows)} companies "
          f"({int((y == 0).sum())} healthy, {int(y.sum())} distress)")

    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=3000))
    oof = cross_val_predict(pipe, X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    print(f"Leave-one-out  PR-AUC {average_precision_score(y, oof):.3f}  "
          f"ROC-AUC {roc_auc_score(y, oof):.3f}")

    pipe.fit(X, y)

    # Persist the fitted model as a plain-JSON artifact the app can serve without sklearn
    # (requirements.txt is kept lean on purpose). We store the medians, the standardisation
    # stats and the logistic coefficients, then verify a pure-numpy forward pass reproduces
    # sklearn's predict_proba exactly before writing anything.
    imp, scaler, logit = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    medians = imp.statistics_
    mean, scale = scaler.mean_, scaler.scale_
    coef, intercept = logit.coef_[0], float(logit.intercept_[0])

    Ximp = np.where(np.isnan(X), medians, X)
    z = (Ximp - mean) / scale @ coef + intercept
    ours = 1.0 / (1.0 + np.exp(-z))
    ref = pipe.predict_proba(X)[:, 1]
    assert np.allclose(ours, ref, atol=1e-9), "numpy forward pass does not match sklearn"
    print("numpy/sklearn parity: OK (max diff "
          f"{np.max(np.abs(ours - ref)):.2e})")

    artifact = {
        "model": "logistic_regression",
        "note": "Altman's four ratios re-fit on Indian companies labelled by real insolvency outcomes. "
                "Probabilities are on the balanced training prior, not the base rate.",
        "features": FEATS,
        "medians": medians.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "n": int(len(y)),
        "n_distress": int(y.sum()),
        "loo_roc_auc": round(float(roc_auc_score(y, oof)), 4),
        "loo_pr_auc": round(float(average_precision_score(y, oof)), 4),
    }
    out = HERE.parents[1] / "models" / "india_logistic.json"
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Saved {out.relative_to(HERE.parents[1])}")

    print("\nTest on the six demo companies (held out of training):")
    npass = 0
    for fin, prior, expect in f.ROSTER:
        fe = f.compute_features(fin, prior=prior)
        x = np.array([[fe.get(c, np.nan) for c in FEATS]], "float64")
        p = float(pipe.predict_proba(x)[0, 1])
        band = f.band_for(p * 100)
        ok = band in f._ACCEPT[expect]
        npass += ok
        print(f"  {fin.company:16} expected {expect:9} -> P={p:.2f}  {band:14} {'OK' if ok else 'MISS'}")
    print(f"\n{npass}/6 correct.")


if __name__ == "__main__":
    main()

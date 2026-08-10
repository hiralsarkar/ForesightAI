"""Research stack: the ML benchmark, tuning and explainability behind the report.

This is the model-building and evaluation code (LightGBM/XGBoost, Optuna, SHAP,
calibration, monotonicity audits) used by the notebooks. It is deliberately kept out
of the shipped engine (src/foresight.py), which only needs Altman, the signals and the
fusion. Nothing here is imported by the live app.
"""

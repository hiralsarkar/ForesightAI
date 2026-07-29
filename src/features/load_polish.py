"""Loader for the Polish Companies Bankruptcy `.arff` files.

The files are ARFF with `?` for missing values and a `{0,1}` nominal target. We parse
them directly rather than depending on `scipy.io.arff`, which returns the target as
bytes and chokes on some of the malformed numeric fields in this dataset.

Horizon semantics matter for the story: `1year.arff` means "bankrupt within 1 year of
the reported financials" -- the shortest, hardest-to-act-on warning. `5year.arff` is
the earliest signal. Foresight AI is about early warning, so the longer horizons are
not throwaways; they are the point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .polish_schema import N_ATTRS, TARGET

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
HORIZONS = (1, 2, 3, 4, 5)


def _arff_path(horizon: int) -> Path:
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    return RAW_DIR / f"{horizon}year.arff"


def load_horizon(horizon: int, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load one forecasting horizon into a DataFrame.

    Returns columns `Attr1..Attr64` as float64 (missing as NaN) plus an int8 `class`.
    A `horizon` column is attached so stacked frames stay traceable.
    """
    path = _arff_path(horizon) if raw_dir is None else Path(raw_dir) / f"{horizon}year.arff"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset from UCI id 365 into {path.parent}."
        )

    rows: list[list[str]] = []
    in_data = False
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            if not line or line.startswith("%"):
                continue
            rows.append(line.split(","))

    if not rows:
        raise ValueError(f"no data rows parsed from {path}")

    width = N_ATTRS + 1
    bad = [i for i, r in enumerate(rows) if len(r) != width]
    if bad:
        raise ValueError(
            f"{path.name}: {len(bad)} rows have wrong field count "
            f"(expected {width}); first at data row {bad[0]}"
        )

    columns = [f"Attr{i}" for i in range(1, N_ATTRS + 1)] + [TARGET]
    df = pd.DataFrame(rows, columns=columns)
    df = df.replace({"?": np.nan, "": np.nan})
    df = df.astype("float64")

    if not df[TARGET].dropna().isin({0.0, 1.0}).all():
        raise ValueError(f"{path.name}: target contains values outside {{0,1}}")

    df[TARGET] = df[TARGET].astype("int8")
    df.insert(0, "horizon", np.int8(horizon))
    return df


def load_all(raw_dir: Path | None = None) -> pd.DataFrame:
    """Stack all five horizons.

    Note: a company may appear across multiple horizon files. Any cross-horizon
    modelling must group by company to avoid leakage -- but the dataset ships no
    company identifier, so horizons are modelled separately by default.
    """
    return pd.concat(
        [load_horizon(h, raw_dir) for h in HORIZONS], ignore_index=True
    )


def class_balance(df: pd.DataFrame) -> dict[str, float]:
    """Distress rate and counts -- the numbers that justify banning accuracy."""
    counts = df[TARGET].value_counts()
    n = int(len(df))
    distress = int(counts.get(1, 0))
    return {
        "n": n,
        "distress": distress,
        "healthy": int(counts.get(0, 0)),
        "distress_rate": round(distress / n, 5) if n else 0.0,
        "majority_baseline_accuracy": round((n - distress) / n, 5) if n else 0.0,
    }


def missingness(df: pd.DataFrame) -> pd.Series:
    """Fraction missing per attribute, descending. Attr37 is the known worst offender."""
    attrs = [c for c in df.columns if c.startswith("Attr")]
    return df[attrs].isna().mean().sort_values(ascending=False)

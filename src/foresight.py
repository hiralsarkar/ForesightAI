"""Foresight AI - the model and scoring code, all in one place.

Data loading, the Altman engine, the distress model (training, tuning, calibration,
SHAP), the four digital signals, the combined score, and the report/narrative helpers.
The notebooks and the app both import from here.
"""

from __future__ import annotations


"""Attribute schema for the Polish Companies Bankruptcy dataset (UCI id 365).

This module is the keystone for two things:

  * Train/serve feature parity -- the model trains on Polish `Attr1..Attr64`
    but must score Indian companies rebuilt from Screener.in financials. `serving`
    records whether each attribute is reproducible on that path.
  * Plain-English SHAP labels -- `label` is what the user sees on
    a waterfall chart. Never surface `Attr14` in the UI.

The `.arff` headers carry no definitions (verified: they are bare `@attribute AttrN
numeric`). Definitions below come from the UCI dataset documentation.

Serving availability is expressed against what Screener.in actually publishes:
  P&L         Sales, Expenses, Operating Profit, Other Income, Interest,
              Depreciation, Profit before tax, Tax%, Net Profit
  Balance     Equity Capital, Reserves, Borrowings, Other Liabilities,
              Total Liabilities, Fixed Assets, CWIP, Investments, Other Assets,
              Total Assets
  Ratios      Debtor Days, Inventory Days, Days Payable, Cash Conversion Cycle,
              Working Capital Days, ROCE%

Screener does not publish current assets / inventory / receivables as balance-sheet
line items, but they are recoverable from the days-ratios (e.g. receivables =
Debtor Days * Sales / 365). Attributes needing that reconstruction are DERIVED.
"""


from dataclasses import dataclass
from enum import Enum
from typing import Literal

TARGET = "class"
N_ATTRS = 64


class Category(str, Enum):
    """Ratio families. The first five mirror the dashboard's grouping."""

    LIQUIDITY = "Liquidity"
    SOLVENCY = "Solvency"
    PROFITABILITY = "Profitability"
    EFFICIENCY = "Efficiency"
    CASH_FLOW = "Cash Flow"
    GROWTH = "Growth"
    SIZE = "Size"


class Serving(str, Enum):
    """Can this attribute be rebuilt for an Indian company from Screener data?"""

    DIRECT = "direct"  # straight from published line items
    DERIVED = "derived"  # needs reconstruction (typically via days-ratios)
    UNAVAILABLE = "unavailable"  # not recoverable; must not enter the served model


@dataclass(frozen=True)
class Attribute:
    id: str  # "Attr1"
    formula: str  # definition as published by UCI
    label: str  # business English, shown in the UI
    category: Category
    serving: Serving


def _a(n: int, formula: str, label: str, category: Category, serving: Serving) -> Attribute:
    return Attribute(f"Attr{n}", formula, label, category, serving)


P = Category.PROFITABILITY
L = Category.LIQUIDITY
S = Category.SOLVENCY
E = Category.EFFICIENCY
C = Category.CASH_FLOW
G = Category.GROWTH
Z = Category.SIZE

D = Serving.DIRECT
V = Serving.DERIVED
U = Serving.UNAVAILABLE

ATTRIBUTES: tuple[Attribute, ...] = (
    _a(1, "net profit / total assets", "Return on Assets", P, D),
    _a(2, "total liabilities / total assets", "Debt to Assets", S, D),
    _a(3, "working capital / total assets", "Working Capital / Total Assets", L, V),
    _a(4, "current assets / short-term liabilities", "Current Ratio", L, V),
    _a(
        5,
        "[(cash + short-term securities + receivables - short-term liabilities) "
        "/ (operating expenses - depreciation)] * 365",
        "Defensive Interval (days of liquidity)",
        L,
        V,
    ),
    _a(6, "retained earnings / total assets", "Retained Earnings / Total Assets", P, D),
    _a(7, "EBIT / total assets", "Operating Return on Assets", P, D),
    _a(8, "book value of equity / total liabilities", "Equity to Debt", S, D),
    _a(9, "sales / total assets", "Asset Turnover", E, D),
    _a(10, "equity / total assets", "Equity Ratio", S, D),
    _a(
        11,
        "(gross profit + extraordinary items + financial expenses) / total assets",
        "Pre-Financing Return on Assets",
        P,
        V,
    ),
    _a(12, "gross profit / short-term liabilities", "Gross Profit / Short-Term Liabilities", P, V),
    _a(13, "(gross profit + depreciation) / sales", "Cash Gross Margin", P, V),
    _a(14, "(gross profit + interest) / total assets", "Pre-Interest Return on Assets", P, V),
    _a(
        15,
        "(total liabilities * 365) / (gross profit + depreciation)",
        "Debt Repayment Period (days)",
        S,
        V,
    ),
    _a(16, "(gross profit + depreciation) / total liabilities", "Cash Flow to Debt", C, V),
    _a(17, "total assets / total liabilities", "Asset Coverage of Debt", S, D),
    _a(18, "gross profit / total assets", "Gross Return on Assets", P, V),
    _a(19, "gross profit / sales", "Gross Margin", P, V),
    _a(20, "(inventory * 365) / sales", "Inventory Days", E, D),
    _a(21, "sales (n) / sales (n-1)", "Sales Growth (year on year)", G, D),
    _a(22, "profit on operating activities / total assets", "Operating Profit / Total Assets", P, D),
    _a(23, "net profit / sales", "Net Profit Margin", P, D),
    _a(24, "gross profit (in 3 years) / total assets", "3-Year Gross Profit / Total Assets", P, V),
    _a(25, "(equity - share capital) / total assets", "Reserves / Total Assets", S, D),
    _a(26, "(net profit + depreciation) / total liabilities", "Cash Earnings to Debt", C, D),
    _a(27, "profit on operating activities / financial expenses", "Interest Coverage Ratio", S, D),
    _a(28, "working capital / fixed assets", "Working Capital / Fixed Assets", L, V),
    _a(29, "logarithm of total assets", "Company Size (log total assets)", Z, D),
    _a(30, "(total liabilities - cash) / sales", "Net Debt to Sales", S, V),
    _a(31, "(gross profit + interest) / sales", "Pre-Interest Margin", P, V),
    _a(32, "(current liabilities * 365) / cost of products sold", "Payable Days", E, V),
    _a(33, "operating expenses / short-term liabilities", "Operating Expenses / Short-Term Liabilities", E, V),
    _a(34, "operating expenses / total liabilities", "Operating Expenses / Total Liabilities", E, V),
    _a(35, "profit on sales / total assets", "Sales Profit / Total Assets", P, V),
    _a(36, "total sales / total assets", "Total Asset Turnover", E, D),
    _a(37, "(current assets - inventories) / long-term liabilities", "Quick Assets / Long-Term Debt", L, V),
    _a(38, "constant capital / total assets", "Permanent Capital / Total Assets", S, V),
    _a(39, "profit on sales / sales", "Sales Profit Margin", P, V),
    _a(
        40,
        "(current assets - inventory - receivables) / short-term liabilities",
        "Cash Ratio",
        L,
        V,
    ),
    _a(41, "total liabilities / ((profit on operating activities + depreciation) * (12/365))", "Debt to Annualised Operating Cash Flow", S, V),
    _a(42, "profit on operating activities / sales", "Operating Margin", P, D),
    _a(43, "rotation receivables + inventory turnover in days", "Cash Conversion Days", E, D),
    _a(44, "(receivables * 365) / sales", "Receivable Days", E, D),
    _a(45, "net profit / inventory", "Net Profit / Inventory", E, V),
    _a(46, "(current assets - inventory) / short-term liabilities", "Quick Ratio", L, V),
    _a(47, "(inventory * 365) / cost of products sold", "Inventory Days (on cost)", E, V),
    _a(48, "EBITDA / total assets", "EBITDA Return on Assets", P, D),
    _a(49, "EBITDA / sales", "EBITDA Margin", P, D),
    _a(50, "current assets / total liabilities", "Current Assets / Total Debt", L, V),
    _a(51, "short-term liabilities / total assets", "Short-Term Debt / Total Assets", S, V),
    _a(52, "(short-term liabilities * 365) / cost of products sold", "Short-Term Debt Coverage (days of cost)", E, V),
    _a(53, "equity / fixed assets", "Equity / Fixed Assets", S, D),
    _a(54, "constant capital / fixed assets", "Permanent Capital / Fixed Assets", S, V),
    _a(55, "working capital", "Working Capital (absolute)", L, V),
    _a(56, "(sales - cost of products sold) / sales", "Gross Margin on Cost", P, V),
    _a(
        57,
        "(current assets - inventory - short-term liabilities) "
        "/ (sales - gross profit - depreciation)",
        "Net Quick Assets / Cash Costs",
        L,
        V,
    ),
    _a(58, "total costs / total sales", "Cost to Sales Ratio", E, D),
    _a(59, "long-term liabilities / equity", "Long-Term Debt to Equity", S, V),
    _a(60, "sales / inventory", "Inventory Turnover", E, V),
    _a(61, "sales / receivables", "Receivables Turnover", E, D),
    _a(62, "(short-term liabilities * 365) / sales", "Short-Term Debt Days", E, V),
    _a(63, "sales / short-term liabilities", "Sales / Short-Term Liabilities", E, V),
    _a(64, "sales / fixed assets", "Fixed Asset Turnover", E, D),
)

BY_ID: dict[str, Attribute] = {a.id: a for a in ATTRIBUTES}

#: SHAP/plot rename map. Use everywhere a feature name reaches a chart or the UI.
LABELS: dict[str, str] = {a.id: a.label for a in ATTRIBUTES}

#: Attribute 55 is an absolute currency amount, not a ratio. It does not transfer
#: across economies (PLN balance sheets vs INR) and is excluded from the served
#: feature set for scale reasons even though it is technically reconstructible.
SCALE_DEPENDENT: frozenset[str] = frozenset({"Attr55"})


def label_for(attr_id: str) -> str:
    """Business-English label for a feature id, falling back to the id itself."""
    return LABELS.get(attr_id, attr_id)


def rename_for_display(names) -> list[str]:
    """Map a sequence of feature ids to display labels (for SHAP plots)."""
    return [label_for(n) for n in names]


def by_category(category: Category) -> tuple[Attribute, ...]:
    return tuple(a for a in ATTRIBUTES if a.category is category)


def serving_feature_set(include_derived: bool = True) -> tuple[str, ...]:
    """Attributes usable by a model that must also score Indian companies.

    This is the feature-parity guard. Training on the full 64 yields a model that cannot
    score the demo companies at all -- a different feature space. Pass
    ``include_derived=False`` for the conservative subset that needs no
    days-ratio reconstruction.
    """
    allowed = {Serving.DIRECT} | ({Serving.DERIVED} if include_derived else set())
    return tuple(
        a.id
        for a in ATTRIBUTES
        if a.serving in allowed and a.id not in SCALE_DEPENDENT
    )


def coverage_summary() -> dict[str, int]:
    """Counts per serving class, for the model card and README."""
    out = {s.value: 0 for s in Serving}
    for a in ATTRIBUTES:
        out[a.serving.value] += 1
    return out


# --------------------------------------------------------------------------------
# Economic direction, for monotonic model constraints.
# --------------------------------------------------------------------------------
# Sign convention matches XGBoost/LightGBM `monotone_constraints`, expressed against
# the model output P(distress):
#
#   -1  higher feature value -> LOWER distress probability   (e.g. interest coverage)
#   +1  higher feature value -> HIGHER distress probability  (e.g. debt / assets)
#    0  unconstrained
#
# We constrain only where the economics are unambiguous. Ratios whose direction is
# genuinely industry-dependent are left at 0 rather than guessed: a wrong constraint
# is worse than no constraint, because it forces the model to encode false economics
# and cannot be detected by looking at the score.
#
# Deliberately left UNCONSTRAINED, with reasons:
#   * Turnover ratios (Attr9, 36, 60, 61, 64) -- asset intensity is a sector fact, not
#     a health fact. A utility and a software firm sit at opposite ends while both
#     being solvent.
#   * Days ratios (Attr20, 32, 43, 44, 47, 52, 62) -- extremes are bad in both
#     directions, and normal ranges differ by orders of magnitude across sectors.
#   * Attr21 sales growth -- decline signals distress, but so does hypergrowth. Byju's
#     is in our own demo set precisely as a growth-then-collapse case, so constraining
#     "more growth = safer" would contradict a case study we intend to show.
#   * Attr29 company size -- large firms fail too. IL&FS was among the largest NBFCs
#     in India when it collapsed.

class Direction(int, Enum):
    RISK_DECREASING = -1
    UNCONSTRAINED = 0
    RISK_INCREASING = 1


_RISK_DECREASING = {
    # Profitability -- more profit is unambiguously safer.
    1, 6, 7, 11, 12, 13, 14, 18, 19, 22, 23, 24, 31, 35, 39, 42, 48, 49, 56,
    # Liquidity -- more cover for near-term obligations is safer.
    3, 4, 5, 28, 37, 40, 46, 50, 55, 57,
    # Solvency / capital structure -- more equity and asset cover is safer.
    8, 10, 17, 25, 38, 53, 54,
    # Cash flow / debt service -- more cash per unit of debt is safer.
    16, 26, 27,
}

_RISK_INCREASING = {
    2,   # total liabilities / total assets
    15,  # debt repayment period (days)
    30,  # net debt / sales
    41,  # debt / annualised operating cash flow
    51,  # short-term liabilities / total assets
    58,  # total costs / total sales
    59,  # long-term liabilities / equity
}

DIRECTIONS: dict[str, Direction] = {
    f"Attr{i}": (
        Direction.RISK_DECREASING
        if i in _RISK_DECREASING
        else Direction.RISK_INCREASING
        if i in _RISK_INCREASING
        else Direction.UNCONSTRAINED
    )
    for i in range(1, N_ATTRS + 1)
}


#: Metrics exposed to the user as sliders, cards, or what-if inputs.
#:
#: Monotonicity is fundamentally a **UI guarantee**, not a global modelling goal: it
#: matters for a metric precisely when a user can move it and watch the score respond.
#: Constraining only these ten costs 0.023 PR-AUC; constraining all 45 directional
#: attributes costs 0.10 for no additional demo safety. Scope the constraint to the
#: surface the user can actually touch.
#:
#: **Anything added to a what-if or stress slider must be added here**,
#: or that control can move the score the wrong way on stage.
SLIDER_FEATURES: tuple[str, ...] = (
    "Attr27",  # Interest Coverage Ratio
    "Attr2",   # Debt to Assets
    "Attr4",   # Current Ratio
    "Attr46",  # Quick Ratio
    "Attr1",   # Return on Assets
    "Attr23",  # Net Profit Margin
    "Attr10",  # Equity Ratio
    "Attr59",  # Long-Term Debt to Equity
    "Attr7",   # Operating Return on Assets
    "Attr17",  # Asset Coverage of Debt
)

Scope = Literal["slider", "all", "none"]


def direction_for(attr_id: str) -> Direction:
    return DIRECTIONS.get(attr_id, Direction.UNCONSTRAINED)


def monotone_vector(features, scope: Scope = "slider") -> tuple[int, ...]:
    """Constraint vector aligned to `features`, for `monotone_constraints`.

    **Order is load-bearing.** Both boosters match this vector positionally against the
    training matrix, so it must be built from the exact feature list passed to `fit`.
    A misaligned vector applies the wrong constraint to the wrong ratio and fails
    silently -- the model still trains, it just encodes nonsense.

    `scope="slider"` (default) constrains only user-facing metrics -- the measured best
    trade. `"all"` constrains every directional attribute; `"none"` disables.
    """
    if scope == "none":
        return tuple(0 for _ in features)
    if scope == "all":
        return tuple(int(direction_for(f)) for f in features)
    if scope == "slider":
        return tuple(
            int(direction_for(f)) if f in SLIDER_FEATURES else 0 for f in features
        )
    raise ValueError(f"scope must be 'slider', 'all', or 'none'; got {scope!r}")


def constraint_summary() -> dict[str, int]:
    counts = {d.name.lower(): 0 for d in Direction}
    for d in DIRECTIONS.values():
        counts[d.name.lower()] += 1
    return counts


def validate_schema() -> None:
    """Fail loudly if the schema drifts from the dataset's real shape."""
    if len(ATTRIBUTES) != N_ATTRS:
        raise ValueError(f"expected {N_ATTRS} attributes, found {len(ATTRIBUTES)}")

    overlap = _RISK_DECREASING & _RISK_INCREASING
    if overlap:
        raise ValueError(f"attributes given contradictory directions: {sorted(overlap)}")

    stray = (_RISK_DECREASING | _RISK_INCREASING) - set(range(1, N_ATTRS + 1))
    if stray:
        raise ValueError(f"direction assigned to non-existent attributes: {sorted(stray)}")

    # A slider metric with no direction would be silently left unconstrained -- the
    # exact failure this whole mechanism exists to prevent.
    undirected = [f for f in SLIDER_FEATURES if DIRECTIONS.get(f) is Direction.UNCONSTRAINED]
    if undirected:
        raise ValueError(f"slider features lack an economic direction: {undirected}")

    unserved = [f for f in SLIDER_FEATURES if f not in serving_feature_set()]
    if unserved:
        raise ValueError(f"slider features not in the serving set: {unserved}")

    expected = [f"Attr{i}" for i in range(1, N_ATTRS + 1)]
    actual = [a.id for a in ATTRIBUTES]
    if actual != expected:
        missing = set(expected) - set(actual)
        raise ValueError(f"attribute ids not contiguous 1..64; missing={sorted(missing)}")

    if len(set(LABELS.values())) != N_ATTRS:
        seen: dict[str, str] = {}
        for a in ATTRIBUTES:
            if a.label in seen:
                raise ValueError(f"duplicate label {a.label!r}: {seen[a.label]} and {a.id}")
            seen[a.label] = a.id


validate_schema()

"""Loader for the Polish Companies Bankruptcy `.arff` files.

The files are ARFF with `?` for missing values and a `{0,1}` nominal target. We parse
them directly rather than depending on `scipy.io.arff`, which returns the target as
bytes and chokes on some of the malformed numeric fields in this dataset.

Horizon semantics matter for the story: `1year.arff` means "bankrupt within 1 year of
the reported financials" -- the shortest, hardest-to-act-on warning. `5year.arff` is
the earliest signal. Foresight AI is about early warning, so the longer horizons are
not throwaways; they are the point.
"""


from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
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

"""Altman Z-Score, computed on the correct variant for this dataset.

The textbook thresholds 1.81 / 2.99 belong to the
**original 1968 Z**, whose X4 term is *market* value of equity / total liabilities.
The Polish companies are effectively private -- there is no market capitalisation in
the data, and `Attr8` is explicitly *book* value of equity / total liabilities. Using
book equity with original-Z cutoffs would silently mis-band every company.

We therefore implement the two private-firm revisions and default to Z'':

    Z'  (1983, private manufacturing, 5 variables)
        0.717 X1 + 0.847 X2 + 3.107 X3 + 0.420 X4 + 0.998 X5
        distress < 1.23        grey 1.23-2.90      safe > 2.90

    Z'' (1995, private, non-manufacturer-neutral, 4 variables; drops sales/TA)
        6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4
        distress < 1.10        grey 1.10-2.60      safe > 2.60

Z'' is the default because the Polish dataset spans sectors and Z'' deliberately drops
the asset-turnover term that makes Z' industry-sensitive.

Component mapping to the dataset (exact, no proxies needed):

    X1 working capital / total assets            Attr3
    X2 retained earnings / total assets          Attr6
    X3 EBIT / total assets                       Attr7
    X4 book value of equity / total liabilities  Attr8
    X5 sales / total assets                      Attr9   (Z' only)

We use Z-double-prime, the private-firm variant, because these are unlisted companies
with no market value of equity. Its published 1.1 and 2.6 cutoffs replace the original
1.81 and 2.99, which only apply to the market-value formulation.
"""


from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Variant = Literal["z2", "zprime"]

ZONE_DISTRESS = "Distress"
ZONE_GREY = "Grey"
ZONE_SAFE = "Safe"

#: Three-class distress labels. Ordered worst to best.
CLASS_DISTRESS = "distress"
CLASS_WATCH = "watch"
CLASS_HEALTHY = "healthy"


@dataclass(frozen=True)
class ZSpec:
    name: str
    coefficients: dict[str, float]
    distress_below: float
    safe_above: float
    description: str


Z_DOUBLE_PRIME = ZSpec(
    name="Z''",
    coefficients={"Attr3": 6.56, "Attr6": 3.26, "Attr7": 6.72, "Attr8": 1.05},
    distress_below=1.10,
    safe_above=2.60,
    description="Altman Z'' (1995) - private firms, sector-neutral, book equity",
)

Z_PRIME = ZSpec(
    name="Z'",
    coefficients={
        "Attr3": 0.717,
        "Attr6": 0.847,
        "Attr7": 3.107,
        "Attr8": 0.420,
        "Attr9": 0.998,
    },
    distress_below=1.23,
    safe_above=2.90,
    description="Altman Z' (1983) - private manufacturing, book equity",
)

SPECS: dict[str, ZSpec] = {"z2": Z_DOUBLE_PRIME, "zprime": Z_PRIME}


def get_spec(variant: Variant = "z2") -> ZSpec:
    try:
        return SPECS[variant]
    except KeyError:
        raise ValueError(f"variant must be one of {sorted(SPECS)}, got {variant!r}") from None


def altman_z(df: pd.DataFrame, variant: Variant = "z2") -> pd.Series:
    """Compute the Altman Z-Score for each row.

    Rows missing any component yield NaN rather than a partial score -- a Z built from
    three of four terms is not a Z, and quietly zero-filling would push companies
    toward the distress band for a data reason rather than a financial one.
    """
    spec = get_spec(variant)
    missing = [c for c in spec.coefficients if c not in df.columns]
    if missing:
        raise KeyError(f"{spec.name} requires missing columns: {missing}")

    components = df[list(spec.coefficients)].astype("float64")
    # A row with any NaN component must not produce a score.
    valid = components.notna().all(axis=1)

    weights = pd.Series(spec.coefficients, dtype="float64")
    z = components.mul(weights, axis=1).sum(axis=1)
    z = z.where(valid, np.nan)
    z.name = f"altman_{variant}"
    return z


def zone(z: pd.Series, variant: Variant = "z2") -> pd.Series:
    """Band a Z-Score series into Distress / Grey / Safe."""
    spec = get_spec(variant)
    out = pd.Series(pd.NA, index=z.index, dtype="object")
    out[z < spec.distress_below] = ZONE_DISTRESS
    out[(z >= spec.distress_below) & (z <= spec.safe_above)] = ZONE_GREY
    out[z > spec.safe_above] = ZONE_SAFE
    out.name = "altman_zone"
    return out


def three_class_label(
    df: pd.DataFrame,
    variant: Variant = "z2",
    target: str = "class",
) -> pd.Series:
    """Nuanced distress label: healthy / watch / distress.

    the design calls for three classes rather than binary bankruptcy, because banks
    think in watchlists, not binary outcomes. The rule is deliberately conservative:

      * an actual bankruptcy in the dataset is always `distress` -- the ground-truth
        label outranks the ratio-based heuristic and is never softened;
      * a surviving company in the Z distress or grey band is `watch`;
      * everything else is `healthy`.

    Companies with no computable Z fall back to their binary label so the column is
    never NaN for a row we intend to train on.
    """
    z = altman_z(df, variant)
    bands = zone(z, variant)

    label = pd.Series(CLASS_HEALTHY, index=df.index, dtype="object")
    label[bands.isin({ZONE_DISTRESS, ZONE_GREY})] = CLASS_WATCH

    if target in df.columns:
        label[df[target] == 1] = CLASS_DISTRESS

    label.name = "distress_class"
    return label


def attach(
    df: pd.DataFrame,
    variant: Variant = "z2",
    target: str = "class",
) -> pd.DataFrame:
    """Return a copy with Z-Score, zone, and three-class label attached."""
    out = df.copy()
    out[f"altman_{variant}"] = altman_z(df, variant)
    out["altman_zone"] = zone(out[f"altman_{variant}"], variant)
    out["distress_class"] = three_class_label(df, variant, target)
    return out


def benchmark_vs_truth(df: pd.DataFrame, variant: Variant = "z2", target: str = "class") -> dict:
    """How well does the classical Z alone separate the real bankruptcies?

    This is the number the ML model has to beat. Showing it explicitly is the point of
    the "show both" requirement -- the panel shows what the ML adds over the
    textbook benchmark.
    """
    z = altman_z(df, variant)
    spec = get_spec(variant)
    ok = z.notna() & df[target].notna()
    if not ok.any():
        return {"variant": spec.name, "scored": 0}

    flagged = z[ok] < spec.distress_below
    truth = df.loc[ok, target] == 1

    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "variant": spec.name,
        "scored": int(ok.sum()),
        "unscored_missing_components": int((~z.notna()).sum()),
        "flagged_distress": int(flagged.sum()),
        "true_distress": int(truth.sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall)
        else 0.0,
    }


"""Shared data model for the four Digital Pulse signals.

Design decisions locked in here:

* **Same 0-100 risk scale and band language as the financial score.** Higher = worse,
  bands Healthy / Watch / Elevated Risk / Critical. This is what makes fusion a
  plain weighted average later -- financial and digital scores speak one language.

* **Every reading carries a specific explanatory datum**, not a restated metric. Not
  "Sentiment: negative" but "Sentiment declining for 11 weeks;
  68% of coverage mentions 'liquidity concerns'." The datum is a required field.

* **Time-series, not snapshot.** A signal is observed at multiple dates. The latest
  reading feeds the gauges; the trajectory feeds the case-study timeline.
  `SignalSeries` holds the history; `.latest()` is what the gauge shows.

* **Trend is first-class**, computed from the series, because for these signals the
  direction matters more than the level (a company going 300 -> 80 job postings is the
  signal, not the absolute 80).
"""


from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class SignalKind(str, Enum):
    NEWS_SENTIMENT = "News Sentiment"
    LEADERSHIP = "Leadership Stability"
    CREDIT_RATING = "Credit Rating"
    HIRING = "Hiring Activity"
    EMPLOYEE = "Employee Confidence"


class Trend(str, Enum):
    """Direction of travel. Worsening is what a risk desk cares about."""

    IMPROVING = "Improving"
    STABLE = "Stable"
    DETERIORATING = "Deteriorating"


# Reuse the single band definition from the serving financial score so the digital
# gauges and the financial gauge cannot drift apart.
def band_for(risk: float) -> str:
    _bf = band_for

    return _bf(risk)


@dataclass(frozen=True)
class SignalReading:
    """One signal, observed at one date, on the shared risk scale."""

    kind: SignalKind
    as_of: date
    risk_score: float          # 0-100, higher = worse
    label: str                 # signal-specific status, e.g. "Contracting"
    datum: str                 # the specific explanatory sentence (required)
    raw: float = 0.0           # underlying measure (sentiment, exit count, ...)
    #: True for a verified hard event -- a tribunal displacing the board, an auditor
    #: resigning. These are *facts*, not inferences, and must not be averaged away by
    #: softer signals (see `composite.combine`). News tone in particular runs the wrong
    #: way on legal process: "NCLAT stays insolvency admission" reads +0.69 positive to
    #: FinBERT while describing a company in insolvency.
    hard_event: bool = False

    @property
    def band(self) -> str:
        return band_for(self.risk_score)


@dataclass
class SignalSeries:
    """A signal's full observed history for one company."""

    kind: SignalKind
    company: str
    readings: list[SignalReading] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.readings.sort(key=lambda r: r.as_of)

    def latest(self) -> Optional[SignalReading]:
        return self.readings[-1] if self.readings else None

    def as_of(self, when: date) -> Optional[SignalReading]:
        """Point-in-time accessor: the most recent reading at or before `when`.

        This is the Trap-3 guard -- scoring a company as of a past date must never see a
        reading published after it.
        """
        eligible = [r for r in self.readings if r.as_of <= when]
        return eligible[-1] if eligible else None

    def trend(self, window: int = 2) -> Trend:
        """Direction over the last `window` readings, on the risk scale."""
        if len(self.readings) < 2:
            return Trend.STABLE
        recent = self.readings[-window:]
        delta = recent[-1].risk_score - recent[0].risk_score
        if delta > 5:
            return Trend.DETERIORATING
        if delta < -5:
            return Trend.IMPROVING
        return Trend.STABLE


def clamp_score(x: float) -> float:
    return float(min(max(x, 0.0), 100.0))

"""Hiring Activity signal (soft).

Active job-posting count over time (Naukri-style). The
*trend* matters more than the absolute -- a company going 300 -> 80 postings is signalling
contraction regardless of its size. Historical posting counts cannot be scraped after the
fact, so the demo data is illustrative and labelled as such (same basis as Glassdoor).

Point-in-time via `hiring_score_as_of`: only counts observed on or before `when` are used.
"""


from dataclasses import dataclass
from datetime import date



@dataclass(frozen=True)
class HiringObservation:
    observed: date
    active_postings: int


def _pct_change(latest: int, baseline: int) -> float:
    if baseline == 0:
        return 0.0
    return (latest - baseline) / baseline


def hiring_score_as_of(
    observations: list[HiringObservation],
    when: date,
    company: str = "",
) -> SignalReading:
    """Hiring reading as of `when`, from the trend across prior observations."""
    hist = sorted([o for o in observations if o.observed <= when], key=lambda o: o.observed)
    if not hist:
        return SignalReading(
            kind=SignalKind.HIRING, as_of=when, risk_score=40.0,
            label="No data", datum="No hiring data available.", raw=0.0,
        )

    latest = hist[-1]
    baseline = hist[0]
    change = _pct_change(latest.active_postings, baseline.active_postings)

    # Contraction raises risk; expansion lowers it. Anchored so a ~60% contraction lands
    # Elevated and steady/growing hiring lands Healthy.
    risk = clamp_score(35.0 - 200.0 * change)

    if change <= -0.08:
        label = "Contracting"
    elif change >= 0.05:
        label = "Growing"
    else:
        label = "Stable"

    datum = (
        f"Headcount {baseline.active_postings:,} -> {latest.active_postings:,} "
        f"({change:+.0%}) since {baseline.observed.isoformat()}."
    )
    return SignalReading(
        kind=SignalKind.HIRING, as_of=when, risk_score=risk,
        label=label, datum=datum, raw=float(latest.active_postings),
    )

"""Leadership Stability signal from BSE-style corporate filings (anchor).

This is the hardest-evidence of the four signals, and it is
uniquely India-specific: every board-level change, KMP resignation, and auditor change
must be filed with the exchange, dated and public. There is **no selection-bias risk** --
a resignation happened on its filing date or it did not. That makes this the anchor the
softer signals lean on.

The signal is the count of *senior* departures in a trailing window (default 6 months),
weighted by seniority: a CFO or auditor walking out is a sharper distress tell than an
independent director rotating off. More than two senior exits in the window is the red
line, and it maps to the Elevated/Critical threshold.

Point-in-time is intrinsic: `leadership_score_as_of(when)` only counts events filed on or before
`when`, so a 2019 assessment never sees a 2020 filing.
"""


from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum



class Role(str, Enum):
    CEO = "CEO"
    CFO = "CFO"
    MD = "Managing Director"
    CHAIRMAN = "Chairman"
    FOUNDER = "Founder"
    AUDITOR = "Auditor"
    COMPANY_SECRETARY = "Company Secretary"
    WHOLETIME_DIRECTOR = "Whole-Time Director"
    INDEPENDENT_DIRECTOR = "Independent Director"
    BOARD = "Board of Directors"


class EventType(str, Enum):
    RESIGNATION = "resignation"
    APPOINTMENT = "appointment"
    AUDITOR_EXIT = "auditor exit"
    #: Insolvency admission vesting the board's powers in a resolution professional.
    #: The most severe governance event there is -- the board stops governing.
    BOARD_SUSPENSION = "board powers suspended"


#: Seniority weights. A departure's contribution to the risk score. Auditor and CFO
#: exits are weighted highest -- in Indian corporate collapses (IL&FS, many NBFCs) an
#: auditor walking out preceded the public crisis.
_EXIT_WEIGHT: dict[Role, float] = {
    Role.BOARD: 5.0,   # whole board displaced by a resolution professional
    Role.AUDITOR: 2.0,
    Role.CFO: 1.8,
    Role.CEO: 1.6,
    Role.MD: 1.6,
    Role.FOUNDER: 1.5,
    Role.CHAIRMAN: 1.4,
    Role.WHOLETIME_DIRECTOR: 1.1,
    Role.COMPANY_SECRETARY: 0.9,
    Role.INDEPENDENT_DIRECTOR: 0.8,
}

# Points per unit of weighted-exit, tuned so "> 2 exits = red" lands in
# the Elevated band: two average senior exits (weight ~1.5 each = 3.0) -> ~54.
_POINTS_PER_WEIGHT = 18.0
_BASELINE = 6.0  # a stable company still shows minor churn


@dataclass(frozen=True)
class LeadershipEvent:
    company: str
    filing_date: date
    role: Role
    person: str
    event_type: EventType
    #: True only for events confirmed against a real, dated public filing. The anchor's
    #: whole value ("a resignation happened on its date or it did not") depends
    #: on this being real -- an unverified stand-in must not be presented as a filing.
    verified: bool = True

    @property
    def is_exit(self) -> bool:
        return self.event_type in (
            EventType.RESIGNATION, EventType.AUDITOR_EXIT, EventType.BOARD_SUSPENSION)


def _window_exits(
    events: list[LeadershipEvent], when: date, months: int
) -> list[LeadershipEvent]:
    start = when - timedelta(days=int(months * 30.44))
    return [
        e for e in events
        if e.is_exit and start <= e.filing_date <= when
    ]


def _describe(exits: list[LeadershipEvent], months: int) -> str:
    """The specific explanatory datum -- who left, not just how many."""
    if not exits:
        return f"No senior departures filed in the last {months} months."

    # A board suspension is not a "departure" -- describe it as what it is.
    suspensions = [e for e in exits if e.event_type is EventType.BOARD_SUSPENSION]
    if suspensions:
        others = len(exits) - len(suspensions)
        extra = f" Also {others} senior departure{'s' if others > 1 else ''} in the period." if others else ""
        d = suspensions[0].filing_date.isoformat()
        return (f"Board powers suspended and vested in a resolution professional "
                f"({d}) - the most severe governance event on record.{extra}")

    roles = [e.role.value for e in exits]
    # Compress duplicates: "2 Independent Directors" rather than listing each.
    counts: dict[str, int] = {}
    for r in roles:
        counts[r] = counts.get(r, 0) + 1
    parts = [f"{n} {r}{'s' if n > 1 else ''}" if n > 1 else r for r, n in counts.items()]
    joined = ", ".join(parts)
    return f"{len(exits)} senior departures in {months} months: {joined}."


def leadership_score_as_of(
    events: list[LeadershipEvent],
    when: date,
    company: str = "",
    months: int = 12,
) -> SignalReading:
    """Leadership-stability reading as of `when`, counting only prior filings."""
    exits = _window_exits(events, when, months)
    weighted = sum(_EXIT_WEIGHT.get(e.role, 1.0) for e in exits)
    risk = clamp_score(_BASELINE + weighted * _POINTS_PER_WEIGHT)

    # A verified auditor exit or board suspension is a hard governance fact, so the reading
    # itself is floored to the red band: a single such event must read as the warning it is
    # in practice (an auditor walking out has preceded most Indian collapses) rather than the
    # ~42 a lone senior exit would otherwise score. This is what lets a hard event floor the
    # composite when the softer signals disagree. Board suspension already reaches ~96 through
    # its seniority weight; the floor makes that explicit and covers the auditor-exit case.
    if any(e.verified and e.event_type is EventType.BOARD_SUSPENSION for e in exits):
        risk = max(risk, 95.0)
    elif any(e.verified and e.event_type is EventType.AUDITOR_EXIT for e in exits):
        risk = max(risk, 78.0)

    n = len(exits)
    if any(e.event_type is EventType.BOARD_SUSPENSION for e in exits):
        label = "Board displaced"      # not "churn" -- the board stopped governing
    elif n == 0:
        label = "Stable"
    elif n <= 2:
        label = "Some churn"
    else:
        label = "High turnover"  # the "> 2 = red" case

    # A verified tribunal order or auditor exit is a hard fact, flagged so the composite
    # cannot average it away against softer, tone-based signals.
    hard = any(
        e.verified and e.event_type in (EventType.BOARD_SUSPENSION, EventType.AUDITOR_EXIT)
        for e in exits
    )
    return SignalReading(
        kind=SignalKind.LEADERSHIP,
        as_of=when,
        risk_score=risk,
        label=label,
        datum=_describe(exits, months),
        raw=float(n),
        hard_event=hard,
    )

"""Employee Confidence signal (soft).

Glassdoor-style average rating over time. This uses a publicly
available historical dataset; a production version would connect to a
licensed feed. The demo data is illustrative and labelled, and the
signal leans on the rating *trend* (Improving / Stable / Declining), not the absolute.

Point-in-time via `reviews_score_as_of`.
"""


from dataclasses import dataclass
from datetime import date


SOURCE_NOTE = "Source: employee review platform, published rating."


@dataclass(frozen=True)
class RatingObservation:
    observed: date
    rating: float  # 1.0 - 5.0


def reviews_score_as_of(
    observations: list[RatingObservation],
    when: date,
    company: str = "",
) -> SignalReading:
    """Employee-confidence reading as of `when`, from the rating level and trend."""
    hist = sorted([o for o in observations if o.observed <= when], key=lambda o: o.observed)
    if not hist:
        return SignalReading(
            kind=SignalKind.EMPLOYEE, as_of=when, risk_score=40.0,
            label="No data", datum="No employee-review data available.", raw=0.0,
        )

    latest = hist[-1]
    baseline = hist[0]
    delta = latest.rating - baseline.rating

    # Level maps to risk (a 4.2 workforce is confident; a 2.5 is not); the trend adjusts.
    # rating 5 -> ~5 risk, 3 -> ~50, 1 -> ~95.
    level_risk = clamp_score(120.0 - 23.0 * latest.rating)
    risk = clamp_score(level_risk - 20.0 * delta)  # improving trend eases, declining adds

    if delta <= -0.3:
        label = "Declining"
    elif delta >= 0.3:
        label = "Improving"
    else:
        label = "Stable"

    datum = (
        f"Employee rating {baseline.rating:.1f} -> {latest.rating:.1f} "
        f"({delta:+.1f}) since {baseline.observed.isoformat()}."
    )
    return SignalReading(
        kind=SignalKind.EMPLOYEE, as_of=when, risk_score=risk,
        label=label, datum=datum, raw=float(latest.rating),
    )

"""News Sentiment signal.

Built fallback-first, on the "never let the demo break" discipline. A
`SentimentScorer` interface has two implementations:

* `LoughranMcDonaldScorer` -- a no-dependency lexicon scorer using the finance-standard
  Loughran-McDonald word lists. General-purpose sentiment (VADER, etc.) mis-reads
  financial text -- "liability", "aggressive", "debt" are neutral-to-technical in
  finance -- so even the fallback uses the right vocabulary.
* `FinBertScorer` -- ProsusAI/finbert, loaded lazily. The volume here
  (~20 headlines x 8 companies) makes model latency a non-issue. Swapped
  in behind the same interface, so nothing downstream changes.

Headlines are real, dated coverage of the roster companies. The reading combines the
30-day mean polarity with its move against the prior 30 days, but the *level* dominates
the label when coverage is clearly positive or negative -- otherwise a company enjoying
successive ratings upgrades reads as "Deteriorating" on a small dip between two good
months.
"""


import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable


# --- Loughran-McDonald lexicon (curated high-frequency subset) -------------------
# The full master dictionary has ~2,300 negative / ~350 positive terms; this is the
# high-signal subset relevant to distress coverage. Terms are stemmed loosely by
# substring match so "restructuring"/"restructure" both hit.
_LM_NEGATIVE = frozenset({
    "loss", "losses", "decline", "declining", "declined", "default", "defaulted",
    "bankrupt", "bankruptcy", "insolvency", "insolvent", "restructur", "distress",
    "downgrade", "downgraded", "litigation", "lawsuit", "probe", "investigation",
    "fraud", "misstatement", "resign", "resigned", "resignation", "exit", "layoff",
    "layoffs", "shortfall", "deficit", "delinquent", "arrears", "impairment",
    "writedown", "writeoff", "liquidation", "moratorium", "downturn", "slump",
    "plunge", "plummet", "crisis", "distressed", "unpaid", "overdue", "breach",
    "covenant", "downsizing", "grounded", "halt", "suspend", "suspended", "delay",
    "delayed", "cut", "cuts", "weak", "weakness", "concern", "concerns", "warning",
    "risk", "risky", "burden", "strain", "struggle", "struggling", "distraught",
    "nonpayment", "unable", "failure", "failed", "negative", "deteriorat",
})
_LM_POSITIVE = frozenset({
    "profit", "profitable", "growth", "grew", "gain", "gains", "strong", "strength",
    "record", "surge", "surged", "upgrade", "upgraded", "beat", "outperform",
    "expansion", "expanding", "robust", "improve", "improved", "improvement",
    "rebound", "recovery", "recovered", "healthy", "resilient", "milestone",
    "award", "leading", "leader", "efficient", "dividend", "bonus", "success",
    "successful", "win", "winning", "optimistic", "upbeat", "solid",
})


@runtime_checkable
class SentimentScorer(Protocol):
    """Maps one headline to a polarity in [-1, +1] (+1 positive, -1 negative)."""

    def score(self, text: str) -> float: ...

    @property
    def name(self) -> str: ...


class LoughranMcDonaldScorer:
    """No-dependency finance-lexicon scorer. The default, always available."""

    name = "Loughran-McDonald"

    def score(self, text: str) -> float:
        t = text.lower()
        pos = sum(1 for w in _LM_POSITIVE if w in t)
        neg = sum(1 for w in _LM_NEGATIVE if w in t)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)


class FinBertScorer:
    """ProsusAI/finbert, loaded lazily. Requires torch + transformers.

    Kept behind the same interface as the fallback so swapping it in changes nothing
    downstream. Construction does not import torch; the first `score()` does.
    """

    name = "FinBERT"

    def __init__(self, model_id: str = "ProsusAI/finbert") -> None:
        self.model_id = model_id
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline  # heavy import, deferred

            self._pipe = pipeline("sentiment-analysis", model=self.model_id, top_k=None)
        return self._pipe

    def score(self, text: str) -> float:
        pipe = self._ensure()
        scores = {d["label"].lower(): d["score"] for d in pipe(text)[0]}
        # FinBERT emits positive / negative / neutral; polarity is the signed difference.
        return float(scores.get("positive", 0.0) - scores.get("negative", 0.0))


class CachedScorer:
    """Replays FinBERT's polarity from a JSON cache: {headline text: score}.

    Lets the deployed app reproduce FinBERT's exact numbers with no torch/transformers
    installed, the same pre-cache-for-offline pattern the narratives use. A cache miss
    (a headline not seen at build time) falls back to the lexicon, so a new headline
    never crashes the app.
    """

    name = "FinBERT (cached)"

    def __init__(self, table: dict, fallback: "SentimentScorer | None" = None) -> None:
        self._table = table
        self._fallback = fallback or LoughranMcDonaldScorer()

    def score(self, text: str) -> float:
        v = self._table.get(text)
        return float(v) if v is not None else self._fallback.score(text)


_SENTIMENT_CACHE = Path(__file__).resolve().parents[1] / "data" / "demo" / "sentiment_cache.json"


def load_sentiment_cache() -> dict | None:
    """The precomputed FinBERT scores, or None if absent."""
    try:
        return json.loads(_SENTIMENT_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None


def prewarm_sentiment_cache() -> Path:
    """Score every demo headline with FinBERT and persist the table (needs torch)."""

    fb = FinBertScorer()
    table: dict[str, float] = {}
    for company in _DATA.values():
        for h in company.get("headlines", []):
            table[h.text] = round(fb.score(h.text), 6)
    _SENTIMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _SENTIMENT_CACHE.write_text(json.dumps(table, indent=1), encoding="utf-8")
    return _SENTIMENT_CACHE


def default_scorer(prefer_finbert: bool = True) -> SentimentScorer:
    """Best available scorer, resolved cache -> FinBERT -> lexicon.

    Validated (docs/serving_indian_companies.md): FinBERT and the lexicon agree on sign, but the
    lexicon reads several distress headlines *neutral* ("defers loan repayment") because
    they contain no lexicon words, while FinBERT reads them negative from context. So the
    deployed app serves FinBERT's exact scores from the cache (no torch needed); locally,
    live FinBERT is used if the cache is absent; the lexicon is the never-breaks fallback.
    """
    if prefer_finbert:
        cache = load_sentiment_cache()
        if cache:
            return CachedScorer(cache)
        try:
            import importlib.util

            if importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"):
                return FinBertScorer()
        except Exception:
            pass
    return LoughranMcDonaldScorer()


@dataclass(frozen=True)
class Headline:
    published: date
    text: str
    source: str = ""


def _polarity_to_risk(polarity: float) -> float:
    """Map mean polarity [-1,+1] to a 0-100 risk score.

    Anchored so healthy-ish coverage (polarity ~ +0.5) lands Healthy (~18), neutral
    lands mid-Watch (~40), and uniformly negative coverage lands Critical (~85).
    """
    return clamp_score(40.0 - 45.0 * polarity)


def _mean_polarity(headlines: list[Headline], scorer: SentimentScorer,
                   start: date, end: date) -> float | None:
    window = [h for h in headlines if start < h.published <= end]
    if not window:
        return None
    return sum(scorer.score(h.text) for h in window) / len(window)


def sentiment_score_as_of(
    headlines: list[Headline],
    when: date,
    scorer: SentimentScorer | None = None,
    company: str = "",
    window_days: int = 30,
) -> SignalReading:
    """News-sentiment reading as of `when`: 30-day mean polarity + trend vs prior 30.

    Point-in-time: only headlines published on or before `when` are considered.
    """
    scorer = scorer or LoughranMcDonaldScorer()
    prior_start = when - timedelta(days=2 * window_days)
    mid = when - timedelta(days=window_days)

    recent = _mean_polarity(headlines, scorer, mid, when)
    previous = _mean_polarity(headlines, scorer, prior_start, mid)

    if recent is None:
        return SignalReading(
            kind=SignalKind.NEWS_SENTIMENT, as_of=when, risk_score=40.0,
            label="No coverage", datum="No news coverage in the last 30 days.", raw=0.0,
        )

    risk = _polarity_to_risk(recent)

    # Direction, and the specific datum the gauge shows.
    n_recent = len([h for h in headlines if mid < h.published <= when])
    # The level dominates the trend when coverage is clearly positive or clearly negative.
    # Without this, a company with strongly positive coverage (e.g. successive ratings
    # upgrades) gets labelled "Deteriorating" on a small dip between two good months.
    if recent > 0.2:
        label, moved = "Positive", ("less emphatic than" if previous is not None
                                    and recent - previous < -0.15 else "in line with")
    elif recent < -0.2:
        label, moved = "Negative", ("sharply more negative than" if previous is not None
                                    and recent - previous < -0.15 else "in line with")
    elif previous is not None and recent - previous < -0.15:
        label, moved = "Deteriorating", "sharply more negative than"
    elif previous is not None and recent - previous > 0.15:
        label, moved = "Improving", "more positive than"
    else:
        label, moved = "Stable", "broadly in line with"

    tone = "negative" if recent < -0.1 else "positive" if recent > 0.1 else "mixed"
    datum = (
        f"{n_recent} headlines in the last 30 days, tone {tone} "
        f"(polarity {recent:+.2f}); coverage {moved} the prior month."
    )

    return SignalReading(
        kind=SignalKind.NEWS_SENTIMENT, as_of=when, risk_score=risk,
        label=label, datum=datum, raw=float(recent),
    )

"""Credit-rating signal.

A long-term issuer/instrument rating from a recognised agency (CRISIL, ICRA, CARE, S&P,
Fitch, Moody's) is the single most CRO-defensible external signal there is. We map the
latest grade to the shared 0-100 risk scale, read the trend against the prior action
(upgrade / affirm / downgrade), and floor the score as a HARD EVENT when the grade is 'D'
(default) - a rating agency declaring default is a fact, not an inference, exactly like an
auditor walking out.

Grades are curated per company from public rating-agency press releases - never inferred.
"""

# Base-letter risk anchors on the 0-100 scale. The +/- modifier nudges within the band.
_GRADE_RISK: dict[str, float] = {
    "AAA": 8.0, "AA": 18.0, "A": 30.0, "BBB": 42.0, "BB": 55.0, "B": 68.0, "C": 82.0, "D": 95.0,
}
# Ordinal rank for trend comparison (higher rank = stronger credit).
_GRADE_RANK = {g: i for i, g in enumerate(("D", "C", "B", "BB", "BBB", "A", "AA", "AAA"))}


@dataclass(frozen=True)
class CreditRatingObservation:
    """One dated rating action. `grade` is the base letter class (e.g. 'AA', 'BB', 'D');
    `notch` is the +/-/'' modifier; `agency` and `scale` ('domestic'/'international') label it."""

    action_date: date
    grade: str
    agency: str
    notch: str = ""            # "+", "-", or ""
    scale: str = "domestic"    # domestic long-term unless noted


def _grade_risk(grade: str, notch: str) -> float:
    base = _GRADE_RISK.get(grade.upper(), 50.0)
    adj = -4.0 if notch == "+" else 4.0 if notch == "-" else 0.0
    return clamp_score(base + adj)


def credit_rating_score_as_of(
    events: list[CreditRatingObservation],
    when: date,
    company: str = "",
) -> SignalReading | None:
    """Latest rating on or before `when`, plus the direction of the last action.

    Returns None when there is no rated history - the composite then renormalises, so an
    unrated company is not penalised for the gap.
    """
    history = sorted((e for e in events if e.action_date <= when), key=lambda e: e.action_date)
    if not history:
        return None
    latest = history[-1]
    risk = _grade_risk(latest.grade, latest.notch)
    is_default = latest.grade.upper() == "D"

    # Trend from the prior distinct grade.
    prior = next((e for e in reversed(history[:-1])
                  if (e.grade, e.notch) != (latest.grade, latest.notch)), None)
    _notch = {"+": 1, "": 0, "-": -1}

    def _fine(e: CreditRatingObservation) -> int:
        return _GRADE_RANK.get(e.grade.upper(), 3) * 3 + _notch.get(e.notch, 0)

    if prior is None:
        label, moved = "Affirmed", "no prior action on record"
    else:
        now_rank = _fine(latest)
        was_rank = _fine(prior)
        if now_rank > was_rank:
            label, moved = "Upgraded", f"up from {prior.grade}{prior.notch} ({prior.agency})"
        elif now_rank < was_rank:
            label, moved = "Downgraded", f"down from {prior.grade}{prior.notch} ({prior.agency})"
        else:
            label, moved = "Affirmed", f"held at {prior.grade}{prior.notch}"

    grade_str = f"{latest.grade}{latest.notch}"
    band = "investment grade" if _GRADE_RANK.get(latest.grade.upper(), 0) >= _GRADE_RANK["BBB"] \
        else "sub-investment (junk) grade"
    if is_default:
        datum = (f"{latest.agency} rates {company or 'the issuer'} at 'D' - default. A hard event: "
                 "the agency is confirming missed payment, not forecasting risk.")
        label = "Default"
    else:
        datum = (f"Latest: {latest.agency} {grade_str} ({latest.scale} long-term), {band}; "
                 f"last action {moved}.")

    return SignalReading(
        kind=SignalKind.CREDIT_RATING, as_of=when, risk_score=risk,
        label=label, datum=datum, raw=float(_GRADE_RANK.get(latest.grade.upper(), 0)),
        hard_event=is_default,
    )


"""Digital Pulse composite.

Fuses the four signal readings into one 0-100 digital risk score on the same scale and
band language as the financial score, so that fusing financial and digital is a
plain weighted average later.

**Scope boundary:** this module stops at the digital composite. It does NOT combine with
the financial score -- that fusion is a later step. Building it here would be drift.

Weights reflect signal reliability: leadership
(dated public filings, no selection-bias risk) and news sentiment carry the signal;
hiring and employee reviews are soft and illustrative, so they inform but do not drive.
Missing signals are dropped and the remaining weights renormalised, so a company with no
review data is not penalised for the gap.
"""


from dataclasses import dataclass, field
from datetime import date
from typing import Optional


_WEIGHTS: dict[SignalKind, float] = {
    SignalKind.CREDIT_RATING: 0.30,   # most CRO-defensible external signal
    SignalKind.LEADERSHIP: 0.25,      # dated public filings, no selection bias
    SignalKind.NEWS_SENTIMENT: 0.25,  # can move before the accounts
    SignalKind.HIRING: 0.10,          # soft, illustrative
    SignalKind.EMPLOYEE: 0.10,        # soft, illustrative
}


@dataclass
class DigitalPulse:
    """The four gauges plus their composite, as of one date."""

    company: str
    as_of: date
    readings: list[SignalReading] = field(default_factory=list)
    composite_score: float = 0.0

    @property
    def band(self) -> str:
        return band_for(self.composite_score)

    def by_kind(self, kind: SignalKind) -> Optional[SignalReading]:
        for r in self.readings:
            if r.kind is kind:
                return r
        return None

    def top_concern(self) -> Optional[SignalReading]:
        """The single most alarming signal -- what the narrative should lead with."""
        return max(self.readings, key=lambda r: r.risk_score) if self.readings else None


def combine(company: str, as_of: date, readings: list[SignalReading]) -> DigitalPulse:
    """Weighted-average the available signal readings into a digital composite."""
    present = [r for r in readings if r is not None]
    total_w = sum(_WEIGHTS.get(r.kind, 0.0) for r in present)

    if total_w == 0:
        score = 0.0
    else:
        score = clamp_score(
            sum(r.risk_score * _WEIGHTS.get(r.kind, 0.0) for r in present) / total_w
        )

    # Hard-event floor. A verified fact (a tribunal displacing the board, an auditor
    # walking out) cannot be diluted below its own reading by softer signals. Without
    # this, Reliance Comm's board-suspension reading of 96 was averaged against a news
    # score of 25 -- because FinBERT reads "NCLAT stays insolvency admission" as
    # *positive* -- and the company de-escalated from Critical to Elevated purely by
    # adding data. Facts floor inferences; that is standard credit practice for a
    # covenant breach or insolvency filing.
    hard = [r.risk_score for r in present if r.hard_event]
    if hard:
        score = max(score, max(hard))

    return DigitalPulse(
        company=company, as_of=as_of, readings=list(present), composite_score=round(score, 1)
    )

"""Market-intelligence signals for the demo roster -- complete across all four signals.

Every company carries news, leadership, hiring and employee sentiment, drawn from current
public reporting (company results, exchange filings, news coverage, employee-review
platforms). Assessment date is mid-2026 for all names, so the signals and the FY2026
financials describe the same moment.

Headcount figures are as reported by the companies; employee ratings are the published
review-platform scores; leadership events are dated announcements; headlines are real,
dated coverage.
"""


from datetime import date

hiring_score = hiring_score_as_of
leadership_score = leadership_score_as_of
reviews_score = reviews_score_as_of
sentiment_score = sentiment_score_as_of
credit_rating_score = credit_rating_score_as_of

AS_OF = date(2026, 5, 31)

# ================================================================== SPICEJET
# Distress on every axis: auditor resignation, furloughs, unpaid salaries, regulator
# intervention. The auditor exit is a hard event -- it floors the digital score.
SPICEJET = dict(
    leadership=[
        LeadershipEvent("SpiceJet", date(2025, 6, 13), Role.AUDITOR,
                        "Walker Chandiok & Co LLP", EventType.AUDITOR_EXIT, verified=True),
    ],
    headlines=[
        Headline(date(2026, 4, 18), "SpiceJet preparing for layoffs as financial troubles intensify", "Business Standard"),
        Headline(date(2026, 5, 20), "SpiceJet employees face months of salary delays as airline seeks emergency funding", "Aviation Today"),
        Headline(date(2026, 6, 9), "SpiceJet delays pilot salaries, seeks government-backed loan to shore up finances", "Business Standard"),
        Headline(date(2026, 6, 22), "SpiceJet seeks emergency funds amid salary delays and shrinking fleet", "The Federal"),
    ],
    hiring=[
        HiringObservation(date(2025, 6, 1), 800),   # engineering workforce
        HiringObservation(date(2026, 4, 1), 738),   # 62 engineers exited, notice waived
        HiringObservation(date(2026, 5, 15), 640),   # six-month furlough from 1 Apr 2026
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.0),
        RatingObservation(date(2026, 5, 15), 2.7),   # published review-platform score
    ],
    # CARE long-term: distress band; upgraded from C to B- after the 2024 fundraise, still junk.
    credit_ratings=[
        CreditRatingObservation(date(2024, 8, 20), "C", "CARE"),
        CreditRatingObservation(date(2025, 6, 1), "B", "CARE", notch="-"),
    ],
)

# =============================================================== OLA ELECTRIC
# A young company burning cash: C-suite exodus, a 5% workforce cut, collapsing share.
OLA = dict(
    leadership=[
        LeadershipEvent("Ola Electric", date(2026, 1, 19), Role.CFO,
                        "Harish Abichandani", EventType.RESIGNATION, verified=True),
        LeadershipEvent("Ola Electric", date(2025, 11, 12), Role.WHOLETIME_DIRECTOR,
                        "Suvonil Chatterjee (CTO)", EventType.RESIGNATION, verified=True),
        LeadershipEvent("Ola Electric", date(2025, 12, 3), Role.WHOLETIME_DIRECTOR,
                        "Anshul Khandelwal (CMO)", EventType.RESIGNATION, verified=True),
    ],
    headlines=[
        Headline(date(2026, 1, 20), "Ola Electric shares tank 5% as CFO Harish Abichandani resigns", "Business Today"),
        Headline(date(2026, 1, 20), "Ola Electric shares drop 7% as CFO resigns; stock slides 24% in 10 sessions", "Business Standard"),
        Headline(date(2026, 1, 31), "Ola Electric lays off 5% staff as EV sales stay below 10,000 units for third consecutive month", "Business Today"),
        Headline(date(2026, 5, 14), "Ola Electric market share slips further as rivals extend lead", "Economic Times"),
    ],
    hiring=[
        HiringObservation(date(2025, 6, 1), 12400),
        HiringObservation(date(2026, 1, 31), 11780),  # 5% cut, ~620 roles
        HiringObservation(date(2026, 5, 15), 11300),
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.1),
        RatingObservation(date(2026, 5, 15), 2.8),
    ],
)

# ============================================================== VODAFONE IDEA
# Negative net worth of ~Rs 35,800 crore behind a one-off accounting profit. Governance
# churn including the head of internal audit; new chairman brought in.
VODAFONE_IDEA = dict(
    leadership=[
        LeadershipEvent("Vodafone Idea", date(2026, 3, 6), Role.COMPANY_SECRETARY,
                        "Gautam Pendse (Head of Internal Audit)", EventType.RESIGNATION, verified=True),
        LeadershipEvent("Vodafone Idea", date(2026, 5, 5), Role.CHAIRMAN,
                        "Ravinder Takkar (stepped down as Chairman)", EventType.RESIGNATION, verified=True),
        LeadershipEvent("Vodafone Idea", date(2026, 6, 20), Role.WHOLETIME_DIRECTOR,
                        "Arvind Nevatia (Chief Enterprise Business Officer)", EventType.RESIGNATION, verified=True),
    ],
    headlines=[
        Headline(date(2026, 5, 5), "Kumar Mangalam Birla takes charge as Vodafone Idea chairman, Ravinder Takkar steps down", "People Matters"),
        Headline(date(2026, 5, 18), "Vodafone Idea appoints M P Sunil Kumar as Chief Enterprise Business Officer as Nevatia exits", "ScanX"),
        Headline(date(2026, 6, 15), "Vodafone Idea carries Rs 80,502 crore AGR liability; subscriber base continues to fall", "Business Standard"),
        Headline(date(2026, 6, 25), "Vodafone Idea trades on funding hopes as government support remains pivotal", "HDFC Sky"),
    ],
    hiring=[
        HiringObservation(date(2025, 6, 1), 9670),
        HiringObservation(date(2026, 5, 15), 9985),   # broadly flat headcount
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.9),
        RatingObservation(date(2026, 5, 15), 3.8),
    ],
    # Deep junk; small upgrade as government equity conversion improved the capital structure.
    credit_ratings=[
        CreditRatingObservation(date(2025, 6, 1), "B", "CARE"),
        CreditRatingObservation(date(2026, 3, 15), "B", "CARE", notch="+"),
    ],
)

# ===================================================================== VEDANTA
# The interesting one: levered financials (grey zone) but every market signal improving --
# four ratings upgrades, a completed demerger, and a top-100 workplace listing.
VEDANTA = dict(
    leadership=[
        LeadershipEvent("Vedanta", date(2026, 6, 1), Role.WHOLETIME_DIRECTOR,
                        "Arun Misra (tenure extended)", EventType.APPOINTMENT, verified=True),
    ],
    headlines=[
        Headline(date(2026, 4, 22), "Fitch upgrades Vedanta Resources to BB- from B+", "Fitch"),
        Headline(date(2026, 5, 14), "S&P Global upgrades Vedanta Resources to BB from B+", "S&P Global"),
        Headline(date(2026, 5, 28), "CRISIL upgrades Vedanta to AA+/Stable; ICRA follows after demerger clarity", "Business Today"),
        Headline(date(2026, 6, 27), "Vedanta ranks in India's top 100 best companies to work for; employees earn Rs 2,500 crore via ESOPs", "Asian Mirror"),
    ],
    hiring=[
        HiringObservation(date(2025, 6, 1), 16870),
        HiringObservation(date(2026, 5, 15), 16498),  # down ~2.2% year on year
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.4),
        RatingObservation(date(2026, 5, 15), 3.5),
    ],
    # The signal-led story: domestic long-term upgraded post-demerger; agencies moving up.
    credit_ratings=[
        CreditRatingObservation(date(2025, 6, 1), "AA", "CRISIL"),
        CreditRatingObservation(date(2026, 5, 28), "AA", "CRISIL", notch="+"),
    ],
)

# ======================================================================= PAYTM
# Recovery story: swung from a Rs 663 crore loss in FY25 to a Rs 552 crore profit in FY26,
# and is hiring again after two years of headcount reduction.
PAYTM = dict(
    leadership=[
        LeadershipEvent("Paytm", date(2026, 2, 10), Role.INDEPENDENT_DIRECTOR,
                        "board appointment", EventType.APPOINTMENT, verified=True),
    ],
    headlines=[
        Headline(date(2026, 4, 30), "Paytm posts third consecutive profitable quarter, revenue up 20% year on year", "Business Standard"),
        Headline(date(2026, 5, 21), "Paytm swings to Rs 552 crore profit in FY26 from Rs 663 crore loss", "Economic Times"),
        Headline(date(2026, 6, 12), "Paytm plans 4,000-person hiring spree for AI push", "Whalesbook"),
    ],
    hiring=[
        HiringObservation(date(2025, 3, 31), 39368),   # after a 10.4% reduction
        HiringObservation(date(2026, 5, 15), 41200),    # rehiring on the AI push
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.1),
        RatingObservation(date(2026, 5, 15), 3.2),
    ],
)

# ========================================================================= TCS
# Healthy financials, but the workforce signal is genuinely negative: headcount fell by
# ~23,500 in FY26 and attrition rose. The platform surfaces that without over-reacting.
TCS = dict(
    leadership=[
        LeadershipEvent("TCS", date(2025, 9, 1), Role.INDEPENDENT_DIRECTOR,
                        "board appointment", EventType.APPOINTMENT, verified=True),
    ],
    headlines=[
        Headline(date(2026, 4, 15), "TCS spends Rs 1,388 crore on restructuring in FY26 as 23,000 roles go, attrition climbs to 13.7%", "Storyboard18"),
        Headline(date(2026, 5, 6), "TCS HR head clarifies headcount drop, says on track to hire 40,000 freshers", "Storyboard18"),
        Headline(date(2026, 6, 18), "TCS rolls out 25,000 campus offers, says layoffs are over", "People Matters"),
    ],
    hiring=[
        HiringObservation(date(2025, 3, 31), 607979),
        HiringObservation(date(2026, 3, 31), 584510),  # down 3.85% year on year
    ],
    ratings=[
        RatingObservation(date(2025, 6, 1), 3.7),
        RatingObservation(date(2026, 5, 15), 3.6),
    ],
    # Top of the scale, rock stable - the healthy anchor.
    credit_ratings=[
        CreditRatingObservation(date(2025, 6, 1), "AAA", "CRISIL"),
        CreditRatingObservation(date(2026, 4, 15), "AAA", "CRISIL"),
    ],
)

_DATA = {
    "SpiceJet": SPICEJET,
    "Ola Electric": OLA,
    "Vodafone Idea": VODAFONE_IDEA,
    "Vedanta": VEDANTA,
    "Paytm": PAYTM,
    "TCS": TCS,
}

DEFAULT_AS_OF = {name: AS_OF for name in _DATA}


def has_signals(company: str) -> bool:
    return company in _DATA


def pulse_as_of(company: str, when: date, scorer: SentimentScorer | None = None) -> DigitalPulse:
    """Assemble the four-signal Digital Pulse for `company` as of `when`."""
    if company not in _DATA:
        raise KeyError(f"no signals for {company!r}; have {sorted(_DATA)}")
    d = _DATA[company]
    readings = [
        leadership_score(d["leadership"], when, company),
        sentiment_score(d["headlines"], when, scorer, company),
        hiring_score(d["hiring"], when, company),
        reviews_score(d["ratings"], when, company),
    ]
    # Credit rating is optional per company; combine() drops None and renormalises weights.
    cr = credit_rating_score(d.get("credit_ratings", []), when, company)
    if cr is not None:
        readings.append(cr)
    return combine(company, when, readings)


def timeline(company: str, dates: list[date], scorer: SentimentScorer | None = None
             ) -> list[DigitalPulse]:
    """Digital Pulse at each date -- the trajectory for a case-study chart."""
    return [pulse_as_of(company, d, scorer) for d in dates]


"""Serving bridge: Screener.in financials -> Polish `AttrN` feature vector.

This is the linchpin of the whole product. The model trains on Polish `Attr1..Attr64`;
this module is what lets it score an *Indian* company. Without it there is no demo.

**What Screener actually gives us (verified against live pages, July 2026).** The
default statements are heavily aggregated -- there is no inventory, receivables, or
current-assets line. So the current-ratio family is not computable from the balance
sheet and comes back NaN. The model routes NaN natively by design, so this
is graceful, but see the structural-missingness warning below.

    P&L         Sales, Expenses, Operating Profit, Other Income, Interest,
                Depreciation, Profit before tax, Net Profit
    Balance     Equity Capital, Reserves, Borrowings, Other Liabilities,
                Total Liabilities (= balance-sheet total), Fixed Assets, CWIP,
                Investments, Other Assets, Total Assets
    Ratios      Debtor Days, Inventory Days, Days Payable, Cash Conversion Cycle,
                Working Capital Days   (a separate section, per-year)

Derived intermediates:
    equity   = Equity Capital + Reserves            (Reserves ~= retained earnings)
    debt     = Borrowings
    ebit     = Profit before tax + Interest
    ebitda   = ebit + Depreciation
    tot_liab = Total Assets - equity                 (all non-equity claims)
    working_capital = Working Capital Days * Sales / 365   (recovers WC without the
                      current-asset breakdown)

**Structural vs informative missingness.**
In training, a missing feature was *informative* -- distressed Polish firms failed to
report. Here, the current-ratio family is missing *structurally* -- Screener never
breaks it out, for healthy and distressed companies alike. If the model learned
"feature missing -> distress" and every Indian firm is missing the same features, every
Indian score gets a systematic upward nudge. This module cannot fix that; it only
surfaces it. The check is `validate_roster()`: the healthy control names (TCS, Paytm)
must land in the healthy band, or the structural-NaN features must be neutralised on the
serving path. Do not trust any Indian score until that passes.
"""


import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd



class Provenance(str, Enum):
    """Where a served feature's value comes from, for coverage auditing."""

    STATEMENT = "statement"      # from the aggregated P&L / balance sheet
    DAYS_RATIO = "days_ratio"    # from Screener's separate ratios section
    STRUCTURAL_NA = "structural" # not computable from Screener's aggregation -> NaN


@dataclass
class ScreenerFinancials:
    """One company-year of Screener data. All amounts in the same currency unit (Cr).

    Point-in-time: the values must be as reported *for that financial year*, not
    restated later. `year` is the March-ending fiscal year.
    """

    company: str
    year: int

    # P&L
    sales: float
    expenses: float
    operating_profit: float
    other_income: float
    interest: float
    depreciation: float
    profit_before_tax: float
    net_profit: float

    # Balance sheet
    equity_capital: float
    reserves: float
    borrowings: float
    other_liabilities: float
    total_assets: float
    fixed_assets: float
    investments: float = 0.0
    cwip: float = 0.0
    other_assets: float = 0.0

    # Ratios section (optional; None -> the dependent AttrN become NaN)
    debtor_days: Optional[float] = None
    inventory_days: Optional[float] = None
    days_payable: Optional[float] = None
    working_capital_days: Optional[float] = None

    # Market value of equity (Rs Cr) for a listed company -> the original 1968 Altman Z.
    market_cap: Optional[float] = None

    ticker: str = ""

    # ---- derived intermediates -------------------------------------------------
    @property
    def equity(self) -> float:
        return self.equity_capital + self.reserves

    @property
    def total_liabilities(self) -> float:
        """All non-equity claims. Screener's own 'Total Liabilities' row is the
        balance-sheet total (== total assets), which is *not* what the Polish
        `total liabilities` attributes mean."""
        return self.total_assets - self.equity

    @property
    def ebit(self) -> float:
        return self.profit_before_tax + self.interest

    @property
    def ebitda(self) -> float:
        return self.ebit + self.depreciation

    @property
    def working_capital(self) -> Optional[float]:
        if self.working_capital_days is None:
            return None
        return self.working_capital_days * self.sales / 365.0


def _safe(numer: float, denom: float) -> float:
    """Ratio with a NaN guard. A zero denominator is undefined, not zero or infinity --
    returning inf would poison the feature and any downstream aggregate."""
    if denom is None or denom == 0 or (isinstance(denom, float) and math.isnan(denom)):
        return float("nan")
    if numer is None or (isinstance(numer, float) and math.isnan(numer)):
        return float("nan")
    return numer / denom


def compute_features(
    fin: ScreenerFinancials,
    prior: Optional[ScreenerFinancials] = None,
    prior3: Optional[list[ScreenerFinancials]] = None,
) -> dict[str, float]:
    """Map one company-year to the serving `AttrN` features.

    Returns a dict over exactly `serving_feature_set()`; features Screener cannot
    support are NaN. Formulas follow `polish_schema` definitions, translated into
    Screener quantities.

    `prior` is the immediately preceding fiscal year, needed for growth features
    (Attr21). `prior3` is up to three preceding years, for the 3-year gross-profit
    feature (Attr24). Both optional -- absent history just yields NaN for those, which
    the model routes natively.
    """
    ta = fin.total_assets
    eq = fin.equity
    tl = fin.total_liabilities
    sales = fin.sales
    wc = fin.working_capital
    op = fin.operating_profit

    # Every serving attribute, computed where Screener allows and NaN otherwise.
    # Grouped by the intermediate they depend on so the provenance is auditable.
    v: dict[str, float] = {
        # --- from the aggregated statements -----------------------------------
        "Attr1": _safe(fin.net_profit, ta),                       # ROA
        "Attr2": _safe(tl, ta),                                   # debt/assets
        "Attr6": _safe(fin.reserves, ta),                         # retained earnings/TA
        "Attr7": _safe(fin.ebit, ta),                             # EBIT/TA
        "Attr8": _safe(eq, tl),                                   # equity/liabilities
        "Attr9": _safe(sales, ta),                                # asset turnover
        "Attr10": _safe(eq, ta),                                  # equity ratio
        "Attr17": _safe(ta, tl),                                  # asset cover of debt
        "Attr22": _safe(fin.operating_profit, ta),               # op profit/TA
        "Attr23": _safe(fin.net_profit, sales),                  # net margin
        "Attr25": _safe(fin.reserves, ta),                       # reserves/TA
        "Attr26": _safe(fin.net_profit + fin.depreciation, tl),  # cash earnings/debt
        "Attr27": _safe(fin.operating_profit, fin.interest),     # interest coverage
        "Attr29": math.log(ta) if ta and ta > 0 else float("nan"),  # log size
        "Attr36": _safe(sales, ta),                              # total asset turnover
        "Attr42": _safe(fin.operating_profit, sales),           # operating margin
        "Attr48": _safe(fin.ebitda, ta),                         # EBITDA/TA
        "Attr49": _safe(fin.ebitda, sales),                     # EBITDA margin
        "Attr53": _safe(eq, fin.fixed_assets),                  # equity/fixed assets
        "Attr58": _safe(sales - fin.operating_profit, sales),   # cost/sales
        "Attr64": _safe(sales, fin.fixed_assets),               # fixed asset turnover
        "Attr34": _safe(fin.expenses, tl),                      # opex / total liabilities
        "Attr13": _safe(op + fin.depreciation, sales),          # cash gross margin
        "Attr41": _safe(tl, (op + fin.depreciation) * (12.0 / 365.0)),  # debt/annualised op CF
        "Attr30": _safe(tl, sales),                             # ~net debt/sales (no cash split)
        "Attr16": _safe(fin.net_profit + fin.depreciation, tl), # cash flow to debt (~Attr26)
        "Attr24": float("nan"),  # filled from prior3 below
        "Attr21": float("nan"),  # filled from prior below
        # gross-profit proxies: Screener has no COGS split, so gross ~= operating.
        "Attr11": _safe(fin.ebit, ta),                          # pre-financing RoA (~EBIT/TA)
        "Attr14": _safe(fin.ebit, ta),                          # pre-interest RoA
        "Attr18": _safe(fin.operating_profit, ta),             # gross RoA (~op/TA)
        "Attr19": _safe(fin.operating_profit, sales),          # gross margin (~OPM)
        "Attr31": _safe(fin.ebit, sales),                      # pre-interest margin
        "Attr35": _safe(fin.operating_profit, ta),             # sales profit/TA
        "Attr39": _safe(fin.operating_profit, sales),          # sales profit margin
        "Attr56": _safe(fin.operating_profit, sales),          # gross margin on cost (~OPM)
        # --- from Working Capital Days ----------------------------------------
        "Attr3": _safe(wc, ta) if wc is not None else float("nan"),   # WC/TA
        "Attr28": _safe(wc, fin.fixed_assets) if wc is not None else float("nan"),
        # --- from the days-ratio section --------------------------------------
        "Attr20": fin.inventory_days if fin.inventory_days is not None else float("nan"),
        "Attr44": fin.debtor_days if fin.debtor_days is not None else float("nan"),
        "Attr47": fin.inventory_days if fin.inventory_days is not None else float("nan"),
        "Attr32": fin.days_payable if fin.days_payable is not None else float("nan"),
        "Attr62": fin.days_payable if fin.days_payable is not None else float("nan"),
        "Attr43": (
            (fin.debtor_days + fin.inventory_days)
            if (fin.debtor_days is not None and fin.inventory_days is not None)
            else float("nan")
        ),
        "Attr60": _safe(365.0, fin.inventory_days) if fin.inventory_days else float("nan"),
        "Attr61": _safe(365.0, fin.debtor_days) if fin.debtor_days else float("nan"),
    }

    # Growth feature (Attr21 = sales_n / sales_{n-1}), from immediate prior year.
    if prior is not None and prior.sales:
        v["Attr21"] = _safe(sales, prior.sales)

    # 3-year gross-profit / total assets (Attr24): cumulative operating profit over the
    # last three years divided by current total assets. Uses whatever history exists.
    if prior3:
        window = [fin] + list(prior3)
        gp3 = sum(r.operating_profit for r in window[:3])
        if len(window) >= 2:  # require at least two years to be meaningful
            v["Attr24"] = _safe(gp3, ta)

    # Everything else in the serving set is structurally unavailable from Screener
    # (current-ratio family, and long/short-term debt split) -> NaN.
    out: dict[str, float] = {}
    for attr in serving_feature_set(include_derived=True):
        out[attr] = float(v.get(attr, float("nan")))
    return out


def compute_with_history(
    history: list[ScreenerFinancials], year: int
) -> dict[str, float]:
    """Compute features for `year`, wiring prior-year context from `history`.

    `history` is a company's records in any order; growth and 3-year features are
    filled from the years preceding `year` where present.
    """
    by_year = {r.year: r for r in history}
    if year not in by_year:
        raise KeyError(f"no record for {year}; have {sorted(by_year)}")
    prior = by_year.get(year - 1)
    prior3 = [by_year[y] for y in (year - 1, year - 2, year - 3) if y in by_year]
    return compute_features(by_year[year], prior=prior, prior3=prior3 or None)


def to_feature_row(
    fin: ScreenerFinancials,
    features: list[str],
    prior: Optional[ScreenerFinancials] = None,
    prior3: Optional[list[ScreenerFinancials]] = None,
) -> np.ndarray:
    """Feature vector aligned to a model's `features` order, NaN-filled off-serving."""
    computed = compute_features(fin, prior=prior, prior3=prior3)
    return np.array([computed.get(f, float("nan")) for f in features], dtype="float64")


def screener_feature_set() -> tuple[str, ...]:
    """The features the bridge can populate given complete inputs + history.

    **This is the true serving feature set** and what the served model must train on.
    Derived empirically from a fully-populated synthetic
    record so it can never drift from what `compute_features` actually assigns: if a
    formula is added or removed above, this set tracks it automatically.
    """
    probe = ScreenerFinancials(
        company="_probe", year=2020,
        sales=1000, expenses=800, operating_profit=200, other_income=20, interest=10,
        depreciation=30, profit_before_tax=210, net_profit=150,
        equity_capital=50, reserves=500, borrowings=200, other_liabilities=250,
        total_assets=1000, fixed_assets=400, investments=100, cwip=20, other_assets=80,
        debtor_days=50, inventory_days=40, days_payable=45, working_capital_days=30,
    )
    prior = ScreenerFinancials(
        company="_probe", year=2019,
        sales=900, expenses=720, operating_profit=180, other_income=18, interest=9,
        depreciation=28, profit_before_tax=189, net_profit=135,
        equity_capital=50, reserves=430, borrowings=210, other_liabilities=240,
        total_assets=930, fixed_assets=390,
    )
    computed = compute_features(probe, prior=prior, prior3=[prior])
    return tuple(a for a in serving_feature_set(include_derived=True)
                 if not math.isnan(computed[a]))


def coverage(fin: ScreenerFinancials) -> dict[str, float]:
    """How much of the serving feature set this company-year actually populates."""
    computed = compute_features(fin)
    serving = serving_feature_set(include_derived=True)
    present = sum(1 for a in serving if not math.isnan(computed[a]))
    return {
        "company": fin.company,
        "year": fin.year,
        "serving_features": len(serving),
        "populated": present,
        "structural_na": len(serving) - present,
        "coverage_pct": round(100 * present / len(serving), 1),
    }


def frame_from_records(records: list[ScreenerFinancials]) -> pd.DataFrame:
    """Build a tidy feature frame (one row per company-year) for batch scoring."""
    rows = []
    for fin in records:
        feats = compute_features(fin)
        rows.append({"company": fin.company, "ticker": fin.ticker, "year": fin.year, **feats})
    return pd.DataFrame(rows)

"""Serving Financial Health Score, anchored on Altman Z''.

The serving work established that the GBM does not transfer to Indian companies while the
linear Altman Z'' does (see `docs/serving_indian_companies.md`). So the score a user sees for an Indian
company is derived from Z'', not from the GBM's calibrated probability.

Two things this module must get right:

1. **A 0-100 risk score monotonic in Z'', anchored on the published zone thresholds** so
   the bands line up exactly with Altman's own Distress / Grey / Safe boundaries. Higher
   score = higher risk (Z'' runs the other way: higher Z'' = safer).

2. **An exact 4-term decomposition** for the waterfall. Because Z'' is linear, each term
   is weight x value and the four terms *sum to the score* with no approximation -- a
   cleaner serving explanation than SHAP on a number we do not display.

The logistic anchoring solves for two fixed points:
    Z'' = 2.60 (safe threshold)     -> risk 25  (Healthy/Watch boundary)
    Z'' = 1.10 (distress threshold) -> risk 50  (Watch/Elevated boundary)
giving center z0 = 1.10 and steepness k = ln(3)/1.5 = 0.732. Everything below the
distress line lands >= 50 (Elevated/Critical); everything above the safe line lands < 25
(Healthy). This makes the gauge bands and the Altman zones the same statement.
"""


import math
from dataclasses import dataclass

import numpy as np


# Logistic anchor constants (derived above; see module docstring).
_Z0 = 1.10
_K = math.log(3.0) / (Z_DOUBLE_PRIME.safe_above - Z_DOUBLE_PRIME.distress_below)  # 0.732

# Human labels for the four Z'' terms, in the language the waterfall shows.
_TERM_LABELS: dict[str, str] = {
    "Attr3": "Working Capital / Total Assets",
    "Attr6": "Retained Earnings / Total Assets",
    "Attr7": "Operating Return on Assets (EBIT/TA)",
    "Attr8": "Book Equity / Total Liabilities",
}


def risk_from_z(z: float) -> float:
    """Map an Altman Z'' to a 0-100 risk score (higher = riskier)."""
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return float("nan")
    return float(100.0 / (1.0 + math.exp(_K * (z - _Z0))))


def z_from_risk(risk: float) -> float:
    """Inverse map -- useful for slider bounds and reporting a target Z''."""
    risk = min(max(risk, 1e-6), 100 - 1e-6)
    return _Z0 + math.log(100.0 / risk - 1.0) / _K


@dataclass(frozen=True)
class ZTerm:
    """One term of the Z'' decomposition."""

    attr: str
    label: str
    coefficient: float
    value: float
    contribution: float  # coefficient * value; exact, sums to Z''

    @property
    def direction(self) -> str:
        # Every Z'' term is risk-*decreasing* in value (higher = safer), so a negative
        # contribution is what pushes toward distress.
        return "reduces risk" if self.contribution > 0 else "increases risk"


@dataclass
class FinancialScore:
    company: str
    year: int
    z_score: float
    risk_score: float          # 0-100
    band: str
    zone: str                  # Altman zone label
    terms: list[ZTerm]
    missing_terms: list[str]   # Z'' components that were unavailable (filled 0)

    @property
    def is_reliable(self) -> bool:
        """A score built with a missing high-weight term (WC/TA at 6.56) is softer."""
        return not self.missing_terms

    def top_drivers(self, n: int = 3) -> list[ZTerm]:
        """Terms pushing hardest toward distress, worst first."""
        return sorted(self.terms, key=lambda t: t.contribution)[:n]


# Bands must match calibrate.BANDS so financial and combined scores speak one language.
_BANDS: tuple[tuple[float, str], ...] = (
    (25.0, "Healthy"),
    (50.0, "Watch"),
    (75.0, "Elevated Risk"),
    (100.1, "Critical"),
)


def band_for(risk: float) -> str:
    if math.isnan(risk):
        return "Unknown"
    for upper, name in _BANDS:
        if risk < upper:
            return name
    return _BANDS[-1][1]


_OZ0, _OK = 2.40, 1.862   # original-Z risk anchoring: Z 2.99 -> 25, 2.40 -> 50, 1.81 -> 75


def risk_from_original_z(z: float) -> float:
    """Map the classic 1968 Altman Z to a 0-100 risk score (higher = riskier)."""
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return float("nan")
    return float(100.0 / (1.0 + math.exp(_OK * (z - _OZ0))))


def original_z(fin: ScreenerFinancials, market_cap: float,
               prior: ScreenerFinancials | None = None):
    """The classic 1968 Altman Z with all five components, using the *market* value of
    equity - available for listed companies, which is why the app can use it live.

        Z = 1.2 A + 1.4 B + 3.3 C + 0.6 D + 1.0 E
        A working capital / assets   B retained earnings / assets   C EBIT / assets
        D market value of equity / total liabilities                E sales / assets

    Returns (z, zone, components); each component is (label, coefficient, value,
    contribution). Zones: Z > 2.99 Safe, 1.81-2.99 Grey, < 1.81 Distress.
    """
    feats = compute_features(fin, prior=prior)
    total_liab = (fin.borrowings or 0.0) + (fin.other_liabilities or 0.0)
    a = feats.get("Attr3", float("nan"))            # working capital / assets
    b = feats.get("Attr6", float("nan"))            # retained earnings / assets
    c = feats.get("Attr7", float("nan"))            # EBIT / assets
    d = (market_cap / total_liab) if total_liab else float("nan")   # market equity / liabilities
    e = (fin.sales / fin.total_assets) if fin.total_assets else float("nan")   # sales / assets
    parts = [("Working capital / assets", 1.2, a),
             ("Retained earnings / assets", 1.4, b),
             ("EBIT / assets", 3.3, c),
             ("Market value equity / liabilities", 0.6, d),
             ("Sales / assets", 1.0, e)]
    z = sum(co * v for _, co, v in parts if v == v)
    zone = "Safe" if z > 2.99 else ("Distress" if z < 1.81 else "Grey")
    return z, zone, [(lbl, co, v, co * v) for lbl, co, v in parts]


def score_company(
    fin: ScreenerFinancials,
    prior: ScreenerFinancials | None = None,
    spec: ZSpec = Z_DOUBLE_PRIME,
    fill_missing: bool = True,
) -> FinancialScore:
    """Compute the Altman-anchored financial score and its exact decomposition.

    `fill_missing=True` substitutes 0 for an unavailable Z'' component (a neutral
    assumption -- e.g. missing working capital -> WC/TA = 0) so the score still computes,
    and records it in `missing_terms`. With `fill_missing=False`, any missing component
    yields a NaN score, matching the strict training-path Altman behaviour.

    For a listed company (`fin.market_cap` set), the *original 1968* Altman Z is used - all
    five components, with the market value of equity - so the whole app speaks one Altman.
    """
    mc = getattr(fin, "market_cap", None)
    if mc is not None and not (isinstance(mc, float) and math.isnan(mc)):
        z, zone_label, comps = original_z(fin, mc, prior=prior)
        risk = risk_from_original_z(z)
        terms = [ZTerm(lbl, lbl, coef, val, (coef * val if val == val else 0.0))
                 for (lbl, coef, val, _cn) in comps]
        missing = [lbl for (lbl, _co, val, _cn) in comps if val != val]
        return FinancialScore(company=fin.company, year=fin.year, z_score=z, risk_score=risk,
                              band=band_for(risk), zone=str(zone_label), terms=terms,
                              missing_terms=missing)

    altman_zone = zone

    feats = compute_features(fin, prior=prior)

    terms: list[ZTerm] = []
    missing: list[str] = []
    z = 0.0
    for attr, coef in spec.coefficients.items():
        val = feats.get(attr, float("nan"))
        if val is None or math.isnan(val):
            missing.append(_TERM_LABELS.get(attr, label_for(attr)))
            if not fill_missing:
                z = float("nan")
            val = 0.0
        contribution = coef * val
        if not math.isnan(z):
            z += contribution
        terms.append(
            ZTerm(attr, _TERM_LABELS.get(attr, label_for(attr)), coef, val, contribution)
        )

    if missing and not fill_missing:
        z = float("nan")

    risk = risk_from_z(z)
    import pandas as pd

    zone_label = (
        altman_zone(pd.Series([z]), "z2").iloc[0] if not math.isnan(z) else "Unknown"
    )

    return FinancialScore(
        company=fin.company,
        year=fin.year,
        z_score=z,
        risk_score=risk,
        band=band_for(risk),
        zone=str(zone_label),
        terms=terms,
        missing_terms=missing,
    )


def narrative(score: FinancialScore) -> str:
    """One-sentence plain-English summary of the financial score's drivers."""
    if math.isnan(score.risk_score):
        return f"{score.company} could not be scored: core balance-sheet terms were missing."

    band = score.band.lower()
    drivers = [t for t in score.top_drivers(3) if t.contribution < 0]
    if drivers:
        driver_text = ", ".join(t.label.split(" (")[0].lower() for t in drivers)
        tail = f"driven mainly by weak {driver_text}"
    else:
        strong = max(score.terms, key=lambda t: t.contribution)
        tail = f"supported by healthy {strong.label.split(' (')[0].lower()}"

    reliability = "" if score.is_reliable else " (note: working-capital data was unavailable)"
    return (
        f"{score.company} carries a {band} financial profile "
        f"(Altman Z'' {score.z_score:.1f}, risk {score.risk_score:.0f}/100), {tail}.{reliability}"
    )

"""Demo roster: current Indian non-financial companies, FY2026 reported financials.

Every company here has **complete data across all four market signals** as well as
financials -- news, leadership, hiring and employee sentiment. That is deliberate: these
are current, actively-covered listed companies, so the signals genuinely exist rather
than having to be reconstructed for a historical date.

All figures are from Screener.in for the year ended March 2026. Non-financial companies
only: a bank's balance sheet is out-of-distribution for an industrial ratio model.

The roster is chosen so the five companies land in *different* places and for *different
reasons* -- three distinct shapes of distress, one levered-but-improving watch case, and
two healthy names with contrasting trajectories.
"""



# ------------------------------------------------------------------ distress
SPICEJET_2026 = ScreenerFinancials(
    "SpiceJet", 2026, sales=5326, expenses=5764, operating_profit=-439, other_income=1442,
    interest=297, depreciation=645, profit_before_tax=62, net_profit=62,
    equity_capital=1413, reserves=-3356, borrowings=4219, other_liabilities=4318,
    total_assets=6594, fixed_assets=2111,
    debtor_days=8, working_capital_days=-297, ticker="SPICEJET",
)
OLA_2026 = ScreenerFinancials(
    "Ola Electric", 2026, sales=2253, expenses=3225, operating_profit=-972, other_income=207,
    interest=380, depreciation=684, profit_before_tax=-1829, net_profit=-1833,
    equity_capital=4411, reserves=-1060, borrowings=2763, other_liabilities=1674,
    total_assets=7788, fixed_assets=3352,
    debtor_days=5, working_capital_days=-185, ticker="OLAELEC",
)
VODAFONE_IDEA_2026 = ScreenerFinancials(
    "Vodafone Idea", 2026, sales=44873, expenses=25870, operating_profit=19003,
    other_income=59148, interest=21495, depreciation=22108,
    profit_before_tax=34548, net_profit=34552,
    equity_capital=108343, reserves=-144101, borrowings=192528, other_liabilities=34868,
    total_assets=191638, fixed_assets=156906,
    debtor_days=16, working_capital_days=-192, ticker="IDEA",
)

# ---------------------------------------------------------------------- watch
VEDANTA_2026 = ScreenerFinancials(
    "Vedanta", 2026, sales=78437, expenses=55254, operating_profit=23183, other_income=14186,
    interest=2817, depreciation=4810, profit_before_tax=29742, net_profit=25096,
    equity_capital=391, reserves=49261, borrowings=32947, other_liabilities=149712,
    total_assets=232311, fixed_assets=30548,
    debtor_days=6, inventory_days=66, working_capital_days=-121, ticker="VEDL",
)

# -------------------------------------------------------------------- healthy
PAYTM_2026 = ScreenerFinancials(
    "Paytm", 2026, sales=8437, expenses=7937, operating_profit=500, other_income=668,
    interest=18, depreciation=568, profit_before_tax=582, net_profit=552,
    equity_capital=64, reserves=15962, borrowings=194, other_liabilities=7695,
    total_assets=23915, fixed_assets=910,
    debtor_days=51, working_capital_days=-55, ticker="PAYTM",
)
TCS_2026 = ScreenerFinancials(
    "TCS", 2026, sales=267021, expenses=194623, operating_profit=72398, other_income=-124,
    interest=1227, depreciation=5560, profit_before_tax=65487, net_profit=49454,
    equity_capital=362, reserves=106878, borrowings=11283, other_liabilities=62644,
    total_assets=181167, fixed_assets=31343,
    debtor_days=93, working_capital_days=38, ticker="TCS",
)

#: Sector labels, defined here so the app UI and the narrative pre-warm use identical
#: strings -- the narrative cache key hashes the prompt, which includes the sector, so a
#: mismatch here silently breaks the offline cache.
SECTORS: dict[str, str] = {
    "SpiceJet": "Airline", "Ola Electric": "Electric Vehicles",
    "Vodafone Idea": "Telecom", "Vedanta": "Metals & Mining",
    "Paytm": "Fintech", "TCS": "IT Services",
}

#: (record, prior-year record or None, expected band family) for the validation gate.
ROSTER: list[tuple[ScreenerFinancials, ScreenerFinancials | None, str]] = [
    (SPICEJET_2026, None, "distress"),
    (OLA_2026, None, "distress"),
    (VODAFONE_IDEA_2026, None, "distress"),
    (VEDANTA_2026, None, "distress"),
    (PAYTM_2026, None, "healthy"),
    (TCS_2026, None, "healthy"),
]

_ACCEPT = {
    "healthy": {"Healthy"},
    "watch": {"Watch", "Elevated Risk"},
    "distress": {"Elevated Risk", "Critical"},
}

# Market value of equity for the listed roster, so scoring uses the original 1968 Altman Z.
_MARKET_CAPS = {"SpiceJet": 1779, "Ola Electric": 17852, "Vodafone Idea": 139654,
                "Vedanta": 106183, "Paytm": 91912, "TCS": 898659}
for _fin, _prior, _expect in ROSTER:
    _fin.market_cap = _MARKET_CAPS.get(_fin.company)


def validate_roster() -> "list[dict]":
    """Score every roster company and check it against its expected band."""

    out = []
    for fin, prior, expect in ROSTER:
        s = score_company(fin, prior=prior)
        out.append({
            "company": s.company, "year": s.year, "expected": expect,
            "z_score": round(s.z_score, 2), "risk": round(s.risk_score, 1),
            "band": s.band, "pass": s.band in _ACCEPT[expect],
        })
    return out

"""Combined Risk Score: fusion of financial and digital signals.

The product's primary output. It fuses the Altman-anchored Financial Score with the
Digital Pulse composite into one 0-100 risk score, and -- crucially -- shows the two legs
separately, because the relationship *between* them is the story.

Two design decisions, both learned the hard way earlier in the build:

1. **Renormalize over available components; never score absent digital as 0.** The current
   roster carries digital signals on every company, but the fusion must stay correct for
   any company that lacks them: a fixed 60/40 with digital=0 would drag every
   financial-only company toward healthy (0.6*fin + 0.4*0) and under-flag a distressed
   one -- the same structural-missingness bias that broke the serving path. So when
   digital is absent, combined = financial, and the UI says so.

2. **No manufactured divergence.** The flagship story is "the gap between
   the legs" (financial healthy, digital weak -> trouble not yet in the numbers). But all
   three dual-signal companies *agree* (TCS both healthy, Jet both critical, Future Retail
   both watch) -- there is no cross-sectional divergence in the real data, and fabricating
   one is exactly the mistake we corrected for Future Retail. So the narrative describes
   what is actually there: agreement (mutual confirmation) or, if it ever arises,
   divergence -- computed from the data, never assumed. The temporal thesis (digital
   leading) lives in the Jet timeline, not here.
"""


from dataclasses import dataclass
from datetime import date
from typing import Optional


DEFAULT_FINANCIAL_WEIGHT = 0.60
DEFAULT_DIGITAL_WEIGHT = 0.40

# Band order, worst last, for judging whether the two legs genuinely disagree. A gap
# within the same band (Jet: financial 100, digital 77 -- both Critical) is corroboration,
# not divergence; only a band-level disagreement is worth flagging.
_BAND_RANK = {"Healthy": 0, "Watch": 1, "Elevated Risk": 2, "Critical": 3}


@dataclass
class CombinedRisk:
    company: str
    as_of: Optional[date]
    financial_score: float
    digital_score: Optional[float]      # None when digital signals are unavailable
    combined_score: float
    financial_weight: float             # actual weight used (renormalized)
    digital_weight: float
    narrative: str

    @property
    def band(self) -> str:
        return band_for(self.combined_score)

    @property
    def has_digital(self) -> bool:
        return self.digital_score is not None

    @property
    def gap(self) -> Optional[float]:
        """digital - financial. Positive => digital is the weaker (riskier) leg."""
        if self.digital_score is None:
            return None
        return round(self.digital_score - self.financial_score, 1)


def _narrative(company: str, fin: float, dig: Optional[float], combined: float) -> str:
    band = band_for(combined).lower()

    if dig is None:
        return (
            f"{company} scores {combined:.0f}/100 ({band}) on financial signals alone. "
            "Digital signals are unavailable for this company, so the combined score "
            "reflects the financial model only."
        )

    fin_rank = _BAND_RANK[band_for(fin)]
    dig_rank = _BAND_RANK[band_for(dig)]

    if fin_rank == dig_rank:
        # The real case for every dual-signal demo company: the legs agree on the band.
        return (
            f"{company} scores {combined:.0f}/100 ({band}). Financial ({fin:.0f}) and "
            f"digital ({dig:.0f}) signals agree, which raises confidence in the assessment."
        )
    if dig_rank > fin_rank:
        # Digital reads a worse band than financials -- the pattern worth surfacing.
        # Computed from the data, never assumed; only fires if it genuinely occurs.
        return (
            f"{company} scores {combined:.0f}/100 ({band}), but digital signals "
            f"({band_for(dig).lower()}, {dig:.0f}) are weaker than the financials "
            f"({band_for(fin).lower()}, {fin:.0f}) suggest. This kind of divergence can "
            "precede financial deterioration -- a signal to look closer."
        )
    return (
        f"{company} scores {combined:.0f}/100 ({band}); the risk shows more in the "
        f"financials ({band_for(fin).lower()}, {fin:.0f}) than in market signals "
        f"({band_for(dig).lower()}, {dig:.0f}) so far."
    )


def fuse(
    financial: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    financial_weight: float = DEFAULT_FINANCIAL_WEIGHT,
    digital_weight: float = DEFAULT_DIGITAL_WEIGHT,
) -> CombinedRisk:
    """Combine the two legs into the primary risk score.

    When `digital` is None the combined score is the financial score and the weights
    collapse to (1.0, 0.0) -- no phantom digital=0 term.
    """
    fin_score = financial.risk_score

    if digital is None:
        combined = fin_score
        w_fin, w_dig, dig_score = 1.0, 0.0, None
    else:
        dig_score = digital.composite_score
        total = financial_weight + digital_weight
        w_fin, w_dig = financial_weight / total, digital_weight / total
        combined = w_fin * fin_score + w_dig * dig_score

    combined = round(combined, 1)
    return CombinedRisk(
        company=financial.company,
        as_of=digital.as_of if digital is not None else None,
        financial_score=round(fin_score, 1),
        digital_score=None if dig_score is None else round(dig_score, 1),
        combined_score=combined,
        financial_weight=round(w_fin, 2),
        digital_weight=round(w_dig, 2),
        narrative=_narrative(financial.company, fin_score, dig_score, combined),
    )


# --- Learned distress probability -------------------------------------------------------
# The comprehensive score above blends a formula (Altman) with dated signals on weights a
# credit desk would recognise. The one place the system *learns* from outcomes is here: a
# logistic regression over Altman's four ratios, re-fit on Indian companies labelled by
# real insolvency outcomes (data/indian/). It is served from a plain-JSON artifact so the
# cloud build needs no
# sklearn; data/indian/train_india_model.py regenerates it and checks numpy/sklearn parity.

def _india_logistic_model():
    import json, functools
    from pathlib import Path
    cache = getattr(_india_logistic_model, "_cache", ...)
    if cache is not ...:
        return cache
    try:
        p = Path(__file__).resolve().parents[1] / "models" / "india_logistic.json"
        cache = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        cache = None
    _india_logistic_model._cache = cache
    return cache


def india_distress_probability(fin: "ScreenerFinancials", prior=None) -> Optional[float]:
    """Learned probability that an Indian company is in distress, 0-1.

    A logistic regression over Altman's four ratios, trained on real Indian insolvency
    outcomes (leave-one-out ROC-AUC 0.97 on 21 companies -- optimistic on a set this small, see
    data/indian/README.md). Returns None if the artifact is missing or any of the four
    ratios cannot be built for this company: we do not median-fill a live company into a
    confident number. Probabilities sit on the balanced training prior, not the base rate.
    """
    import math
    m = _india_logistic_model()
    if m is None:
        return None
    fe = compute_features(fin, prior=prior)
    z = m["intercept"]
    for c, mean, scale, coef in zip(m["features"], m["mean"], m["scale"], m["coef"]):
        v = fe.get(c)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        z += coef * (float(v) - mean) / scale
    return round(1.0 / (1.0 + math.exp(-z)), 4)

"""Macroeconomic Stress Testing.

Maps macro shocks to a company's Altman-anchored risk score through explicit
economic channels. Three macro levers (the ones the problem statement names) plus two
company-specific levers. USD/INR and sector credit spread are deliberately excluded:
USD/INR helps exporters and hurts importers, so a single slider would move IT firms the
wrong way; credit spread just duplicates the interest-rate channel.

**Transmission (each shock feeds a line item; Altman is recomputed once at the end):**

* Interest rate (Δbps, sustained `horizon` years). Interest sits *below* EBIT, so a rate
  shock does not touch WC/TA or EBIT/TA. It raises interest expense - the headline credit
  channel, shown as the change in **interest coverage** - and, sustained over the horizon,
  erodes retained earnings (equity). Modelled on the floating-rate share of borrowings.
* Inflation (Δpp). Compresses the operating margin by a sector-specific pass-through
  factor: Δoperating_profit = −inflation_beta x Δpp x sales. Flows to EBIT/TA and reserves.
* GDP growth (Δpp). A demand shock: Δsales = gdp_beta x Δpp x sales, of which a
  sector-specific incremental margin drops to operating profit.

**Sector elasticities are explicit and rationale'd** - a single global number is
wrong (an airline, a commodity producer and an IT firm have wildly different
pricing power and operating leverage). Defaults are conservative.

Everything is a linear recompute, so the panel updates instantly (<2s trivially).
"""


from dataclasses import dataclass, replace



@dataclass(frozen=True)
class SectorProfile:
    """Macro sensitivities for a sector, each with a one-line rationale."""

    inflation_beta: float   # operating-margin compression per 1pp inflation (frac of sales)
    gdp_beta: float         # sales change per 1pp GDP (%)
    incremental_margin: float  # frac of a sales change that reaches operating profit
    floating_share: float   # share of borrowings that reprices with rates
    rationale: str


# Conservative, sector-differentiated defaults.
_SECTORS: dict[str, SectorProfile] = {
    "Airline": SectorProfile(
        0.010, 1.6, 0.45, 0.70,
        "Thin pricing power and fuel-cost exposure (high inflation hit); highly cyclical, "
        "high fixed cost (demand and operating-leverage sensitive)."),
    "Electric Vehicles": SectorProfile(
        0.010, 1.5, 0.40, 0.70,
        "Imported inputs and price-competitive (high inflation hit); discretionary demand."),
    "Telecom": SectorProfile(
        0.005, 0.5, 0.55, 0.70,
        "Some subscription pricing power (moderate inflation hit); essential service so "
        "demand is defensive, but very high fixed cost amplifies any revenue change."),
    "Metals & Mining": SectorProfile(
        0.003, 1.5, 0.45, 0.60,
        "Commodity producer - output prices broadly track inflation, so input and output "
        "inflation largely offset (low net margin hit); highly cyclical demand."),
    "IT Services": SectorProfile(
        0.003, 0.9, 0.30, 0.50,
        "Strong contractual pricing power and repriceable labour (low inflation hit); "
        "largely variable cost base and moderate cyclicality."),
    "Fintech": SectorProfile(
        0.005, 1.0, 0.35, 0.50,
        "Moderate pricing power and moderate demand cyclicality."),
}
_DEFAULT = SectorProfile(0.006, 1.0, 0.40, 0.60, "Generic non-financial corporate defaults.")


def sector_profile(sector: str) -> SectorProfile:
    return _SECTORS.get(sector, _DEFAULT)


@dataclass(frozen=True)
class Scenario:
    """A stress scenario. All fields default to 'no change'."""

    interest_bps: float = 0.0     # e.g. +200 = rates rise 2 percentage points
    inflation_pp: float = 0.0     # e.g. +3 = inflation 3pp higher
    gdp_pp: float = 0.0           # e.g. -2 = GDP growth 2pp lower
    op_shock_pct: float = 0.0     # company-specific direct hit to operating profit (%)
    leverage_pct: float = 0.0     # company-specific change in borrowings (%)
    horizon_years: int = 3        # interest-rate shocks are sustained this long


@dataclass
class StressResult:
    base_score: float
    stressed_score: float
    base_z: float
    stressed_z: float
    base_coverage: float          # operating profit / interest, before
    stressed_coverage: float      # after
    contributions: dict           # channel -> score delta it alone would cause

    @property
    def delta(self) -> float:
        return round(self.stressed_score - self.base_score, 1)


def _coverage(rec: ScreenerFinancials, extra_interest: float = 0.0) -> float:
    interest = rec.interest + extra_interest
    return float("nan") if interest == 0 else rec.operating_profit / interest


def apply_to_financials(rec: ScreenerFinancials, sc: Scenario, prof: SectorProfile) -> tuple[ScreenerFinancials, float]:
    """Return the shocked financials and the incremental annual interest cost.

    All channels compose on the underlying line items so a combined scenario interacts
    correctly and never double-counts.
    """
    op = rec.operating_profit
    reserves = rec.reserves

    # --- Interest rate: extra interest, sustained over the horizon, erodes equity. ---
    extra_interest = rec.borrowings * prof.floating_share * (sc.interest_bps / 10000.0)
    reserves -= extra_interest * sc.horizon_years

    # --- Inflation: margin compression. ---
    op += -prof.inflation_beta * sc.inflation_pp * rec.sales

    # --- GDP: demand shock through sales -> operating profit at the incremental margin. ---
    d_sales = prof.gdp_beta * (sc.gdp_pp / 100.0) * rec.sales
    op += d_sales * prof.incremental_margin

    # --- Company-specific direct levers. ---
    op += rec.operating_profit * (sc.op_shock_pct / 100.0)
    debt_delta = rec.borrowings * (sc.leverage_pct / 100.0)
    reserves -= debt_delta

    # Operating-profit changes flow to retained earnings for the year.
    reserves += (op - rec.operating_profit)

    shocked = replace(
        rec,
        operating_profit=op,
        profit_before_tax=rec.profit_before_tax + (op - rec.operating_profit) - extra_interest,
        reserves=reserves,
        borrowings=rec.borrowings + debt_delta,
    )
    return shocked, extra_interest


def _score(rec: ScreenerFinancials, prior) -> float:
    import pandas as pd
    z = float(altman_z(pd.DataFrame([compute_features(rec, prior=prior)]), "z2").iloc[0])
    return z


def _z_and_risk(rec: ScreenerFinancials, prior):
    """(z, risk) using the original 1968 Z for a listed company, else the private-firm Z''."""
    mc = getattr(rec, "market_cap", None)
    if mc is not None and not (isinstance(mc, float) and math.isnan(mc)):
        z, _zone, _comps = original_z(rec, mc, prior=prior)
        return z, risk_from_original_z(z)
    z = _score(rec, prior)
    return z, risk_from_z(z)


def run(rec: ScreenerFinancials, sector: str, sc: Scenario, prior=None) -> StressResult:
    """Apply a scenario and return the full before/after picture."""
    prof = sector_profile(sector)

    base_z, base_score = _z_and_risk(rec, prior)
    shocked, extra_int = apply_to_financials(rec, sc, prof)
    stressed_z, stressed_score = _z_and_risk(shocked, prior)

    # Per-channel attribution: what each lever alone would do to the score. Lets the UI
    # say "rising rates account for N of the M-point move".
    channels = {
        "Interest rate": Scenario(interest_bps=sc.interest_bps, horizon_years=sc.horizon_years),
        "Inflation": Scenario(inflation_pp=sc.inflation_pp),
        "GDP growth": Scenario(gdp_pp=sc.gdp_pp),
        "Company-specific": Scenario(op_shock_pct=sc.op_shock_pct, leverage_pct=sc.leverage_pct),
    }
    contributions = {}
    for name, one in channels.items():
        one_shocked, _ = apply_to_financials(rec, one, prof)
        contributions[name] = round(_z_and_risk(one_shocked, prior)[1] - base_score, 1)

    # Coverage after = shocked operating profit / shocked interest.
    new_interest = rec.interest + extra_int
    stressed_cov = (shocked.operating_profit / new_interest) if new_interest else float("nan")

    return StressResult(
        base_score=round(base_score, 1),
        stressed_score=round(stressed_score, 1),
        base_z=round(base_z, 2),
        stressed_z=round(stressed_z, 2),
        base_coverage=round(_coverage(rec), 2),
        stressed_coverage=round(stressed_cov, 2),
        contributions=contributions,
    )


"""LLM provider configuration for the narrative.

The key is read from the environment first, then from a gitignored `secrets.local.json`
at the repo root. It is never hard-coded in committed source (see `.gitignore`).
"""


import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# OpenRouter is OpenAI-compatible. A free instruction-following model is the default;
# override with the OPENROUTER_MODEL env var if a stronger one is available on the key.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# gpt-oss-20b returns clean four-sentence prose; the Nemotron models leak their
# chain-of-thought ("We need to craft...") so they are not used. Gemma is a rate-limited
# fallback. Verified against the live key on 2026-07-21.
DEFAULT_MODEL = "openai/gpt-oss-20b:free"
_FALLBACK_MODELS = (
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
)


def openrouter_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    f = _ROOT / "secrets.local.json"
    if f.exists():
        try:
            return (json.loads(f.read_text(encoding="utf-8")).get("openrouter_api_key") or "").strip() or None
        except Exception:
            return None
    return None


def models() -> tuple[str, ...]:
    override = os.environ.get("OPENROUTER_MODEL")
    if override:
        return (override, *(_m for _m in _FALLBACK_MODELS if _m != override))
    return _FALLBACK_MODELS


def is_configured() -> bool:
    return openrouter_key() is not None

"""AI Financial Narrative.

One paragraph, four sentences, in a credit analyst's voice. Two paths behind one call:

* **LLM path** (used when an API key is configured). The prompt receives *only
  pre-computed numbers* and is instructed never to calculate anything. This is the guard
  against hallucinated financials in a live demo -- the model narrates, it does not do
  arithmetic.
* **Deterministic fallback** (always available). Built from the same structured facts, so
  a missing API key or a timeout degrades the prose, never the correctness. By design:
  "the demo keeps working".

The fallback is written to vary in *substance*, not just in company name: which sentence
is chosen depends on the band, which Altman term dominates, and what the digital signals
say. Test it across companies -- if two outputs read the same, the templates are wrong.
"""


import math
import os
from dataclasses import asdict, dataclass
from typing import Optional




@dataclass
class NarrativeFacts:
    """The structured input. Every number here is already computed -- the LLM only reads."""
    company: str
    sector: str
    combined_score: float
    combined_band: str
    financial_score: float
    financial_band: str
    altman_z: float
    altman_zone: str
    digital_score: Optional[float]
    digital_band: Optional[str]
    top_negative_terms: list[tuple[str, float]]   # (label, contribution) worst first
    top_positive_term: Optional[tuple[str, float]]
    signals: dict[str, str]                        # signal name -> its explanatory datum
    signals_available: bool


def collect_facts(
    combined: CombinedRisk,
    fin: FinancialScore,
    digital: Optional[DigitalPulse],
    sector: str = "",
) -> NarrativeFacts:
    negatives = [(t.label, t.contribution) for t in fin.terms if t.contribution < 0]
    negatives.sort(key=lambda x: x[1])
    positives = [(t.label, t.contribution) for t in fin.terms if t.contribution > 0]
    positives.sort(key=lambda x: -x[1])

    signals: dict[str, str] = {}
    if digital is not None:
        for r in digital.readings:
            signals[r.kind.value] = r.datum

    return NarrativeFacts(
        company=combined.company, sector=sector,
        combined_score=combined.combined_score, combined_band=combined.band,
        financial_score=fin.risk_score, financial_band=fin.band,
        altman_z=fin.z_score, altman_zone=fin.zone,
        digital_score=combined.digital_score,
        digital_band=None if digital is None else digital.band,
        top_negative_terms=negatives[:3], top_positive_term=positives[0] if positives else None,
        signals=signals, signals_available=digital is not None,
    )


# ------------------------------------------------------------------- LLM path
_SYSTEM = (
    "You are a senior credit analyst writing for a bank's risk committee. "
    "Write exactly four sentences in a single paragraph. No headings, no bullet points, "
    "no preamble, no hedging filler. "
    "CRITICAL: every number you need is supplied. Never calculate, infer, estimate or "
    "invent any figure, ratio, date or fact not present in the input. If something is not "
    "given, do not mention it. "
    "Identify the key risk drivers, note any divergence between financial and market "
    "signals, and close with what a lender should monitor going forward. "
    "Write plainly and specifically; do not sound like a chatbot or a template."
)


def _prompt(f: NarrativeFacts) -> str:
    lines = [
        f"Company: {f.company}" + (f" (sector: {f.sector})" if f.sector else ""),
        f"Combined risk score: {f.combined_score:.0f}/100 ({f.combined_band}); higher = riskier.",
        f"Financial score: {f.financial_score:.0f}/100 ({f.financial_band}).",
        f"Altman Z-double-prime: {f.altman_z:.2f} ({f.altman_zone} zone).",
    ]
    if f.digital_score is not None:
        lines.append(f"Digital/market-signal score: {f.digital_score:.0f}/100 ({f.digital_band}).")
    else:
        lines.append("Digital/market signals: NOT AVAILABLE for this company.")
    if f.top_negative_terms:
        drivers = "; ".join(f"{lbl} (contribution {c:+.2f})" for lbl, c in f.top_negative_terms)
        lines.append(f"Largest negative drivers of the financial score: {drivers}.")
    if f.top_positive_term:
        lines.append(f"Strongest supporting factor: {f.top_positive_term[0]} "
                     f"(contribution {f.top_positive_term[1]:+.2f}).")
    for name, datum in f.signals.items():
        lines.append(f"{name}: {datum}")
    return "\n".join(lines)


def _llm(f: NarrativeFacts, timeout: float = 15.0) -> Optional[str]:
    """Try the LLM via OpenRouter (OpenAI-compatible). Any failure returns None so the
    caller falls back silently -- an API problem never degrades the output.

    The prompt carries only pre-computed numbers and the system prompt forbids the model
    from calculating anything, so a weaker free model cannot invent a financial figure.
    """

    key = openrouter_key()
    if not key:
        return None

    # AI prose is optional. A lean Streamlit deployment may not yet have the
    # optional HTTP client installed, which must not stop financial scoring.
    try:
        import httpx
    except ImportError:
        return None

    payload_base = {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _prompt(f)},
        ],
        "max_tokens": 400,
        "temperature": 0.4,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    for model in models():  # try the preferred model, then fallbacks
        try:
            resp = httpx.post(OPENROUTER_URL, headers=headers,
                              json={**payload_base, "model": model}, timeout=timeout)
            if resp.status_code != 200:
                continue
            text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            if text:
                return text
        except Exception:
            continue
    return None


def _chat(system: str, user: str, timeout: float = 15.0, max_tokens: int = 400,
          temperature: float = 0.3) -> Optional[str]:
    """Generic single-turn OpenRouter call. Returns the reply text, or None on any failure
    (no key, network error, non-200, empty) so every caller can fall back silently."""
    key = openrouter_key()
    if not key:
        return None
    try:
        import httpx
    except ImportError:
        return None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    base = {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature}
    for model in models():
        try:
            resp = httpx.post(OPENROUTER_URL, headers=headers,
                              json={**base, "model": model}, timeout=timeout)
            if resp.status_code != 200:
                continue
            text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            if text:
                return text
        except Exception:
            continue
    return None


_NEWS_SENT_SYS = (
    "You are a credit-risk analyst reading news headlines about one listed company. Judge how "
    "much the coverage signals genuine financial DISTRESS, and be strictly magnitude- and "
    "context-aware. A routine market move - a 2-6% share dip, a small earnings miss, a single "
    "analyst's lower target - is NOT distress and should score low. Real distress means "
    "default, insolvency/NCLT, a rating downgrade, fraud or a probe, an auditor or board exit, "
    "large sustained losses, or a steep sustained collapse. Positive or neutral coverage lowers "
    "the risk. Do not over-react to a single dramatic word. Reply with STRICT JSON only."
)


def llm_news_sentiment(company: str, headlines: list[str],
                       timeout: float = 15.0) -> Optional[dict]:
    """Magnitude-aware news distress score via the LLM. Returns
    {risk 0-100, tone -1..1, rationale, cited[]} or None if the LLM is unavailable, so the
    caller keeps the rule-based lexicon score. This is what stops a 4% dip reading like a
    collapse: the model weighs how bad the news actually is, not just that a word appeared."""
    heads = [h for h in (headlines or []) if h][:18]
    if not heads:
        return None
    listing = "\n".join(f"- {h}" for h in heads)
    user = (f"Company: {company}\nRecent headlines:\n{listing}\n\n"
            'Return JSON exactly: {"risk": <int 0-100, higher = more distress>, '
            '"tone": <float -1..1, net tone>, "rationale": "<=25 words", '
            '"cited": ["up to 3 headlines that genuinely signal distress; [] if none"]}')
    text = _chat(_NEWS_SENT_SYS, user, timeout=timeout, max_tokens=320, temperature=0.2)
    if not text:
        return None
    import json, re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        risk = max(0.0, min(100.0, float(d["risk"])))
        tone = max(-1.0, min(1.0, float(d.get("tone", 0.0))))
        cited = [str(x) for x in (d.get("cited") or []) if str(x).strip()][:3]
        return {"risk": round(risk, 1), "tone": round(tone, 2),
                "rationale": str(d.get("rationale", "")).strip()[:240], "cited": cited}
    except Exception:
        return None


_ANALYST_SYS = (
    "You are a credit-risk analyst writing a short assessment of one listed company for a "
    "lending and investment desk. Use ONLY the numbers given - never invent or recompute a "
    "figure. Write 3 to 4 plain sentences: where the company sits on risk, what drives it in "
    "the Altman decomposition, what the recent news adds, and one line on what to watch. No "
    "hype, no hedging filler, no markdown."
)


def live_analyst_summary(company: str, altman_z: float, zone: str, combined_score: float,
                         band: str, term_pairs: list, news_risk: float, news_rationale: str,
                         model_prob, timeout: float = 15.0) -> Optional[str]:
    """LLM analyst summary for a live-scored company, from pre-computed numbers only. Returns
    None if the LLM is unavailable so the caller can fall back to the deterministic narrative."""
    terms = "; ".join(f"{lbl} {c:+.2f}" for lbl, c in term_pairs if c == c) or "n/a"
    mp = "n/a" if model_prob is None else f"{model_prob * 100:.0f}%"
    nr = f"{news_risk:.0f}/100" if news_risk == news_risk else "n/a"
    user = (f"Company: {company}\n"
            f"Combined risk score: {combined_score:.0f}/100 ({band}).\n"
            f"Altman Z (1968): {altman_z:.2f} ({zone}). Altman term contributions: {terms}.\n"
            f"News distress risk: {nr}. News read: {news_rationale or 'no notable coverage'}.\n"
            f"Learned model probability of distress: {mp}.\n"
            "Write the assessment.")
    return _chat(_ANALYST_SYS, user, timeout=timeout, max_tokens=280, temperature=0.35)


# ------------------------------------------------------------ deterministic path
def _band_clause(band: str) -> str:
    return {
        "Healthy": "sits comfortably outside the distress range",
        "Watch": "belongs on the watchlist rather than in the problem book",
        "Elevated Risk": "warrants active management attention now",
        "Critical": "is in genuine financial distress",
    }.get(band, "has been assessed")


def _fallback(f: NarrativeFacts) -> str:
    # 1. Position.
    s1 = (f"{f.company} scores {f.combined_score:.0f} out of 100 ({f.combined_band.lower()}) "
          f"and {_band_clause(f.combined_band)}, with an Altman Z-double-prime of "
          f"{f.altman_z:.1f} placing it in the {f.altman_zone.lower()} zone.")

    # 2. What drives it -- differs by whether the profile is weak or strong.
    if f.top_negative_terms:
        names = [lbl.split(" (")[0].lower() for lbl, _ in f.top_negative_terms]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
        s2 = f"The score is driven primarily by weak {joined}."
    elif f.top_positive_term:
        s2 = (f"No individual balance-sheet term is pulling the score down, with "
              f"{f.top_positive_term[0].split(' (')[0].lower()} the strongest support.")
    else:
        s2 = "No individual balance-sheet term dominates the assessment."

    # 3. Market signals -- corroboration, divergence, or absence.
    if not f.signals_available:
        s3 = ("Market-intelligence signals are not available for this company, so the "
              "assessment rests on reported financials alone.")
    else:
        # Use the SAME band-disagreement test as the combined narrative. Keying
        # this off a numeric gap instead let the two panels contradict each other on
        # screen -- one calling it a divergence while the other called it corroboration.

        fin_rank = _BAND_RANK.get(f.financial_band, 0)
        dig_rank = _BAND_RANK.get(f.digital_band or f.financial_band, 0)
        concern = max(f.signals.items(), key=lambda kv: len(kv[1])) if f.signals else None
        if dig_rank > fin_rank:
            s3 = (f"Market signals are weaker than the financials suggest "
                  f"({f.digital_band.lower()} at {f.digital_score:.0f} against a "
                  f"{f.financial_band.lower()} {f.financial_score:.0f})"
                  + (f"; {concern[0].lower()} is the sharpest of them." if concern else "."))
        elif dig_rank < fin_rank:
            s3 = (f"Market signals ({f.digital_score:.0f}) read better than the financials "
                  f"({f.financial_score:.0f}), so the strain is currently visible in the "
                  "numbers more than in external indicators.")
        else:
            s3 = (f"Market signals agree with the financial picture at "
                  f"{f.digital_score:.0f}, which raises confidence in the assessment.")

    # 4. What to monitor -- band-specific, actionable.
    s4 = {
        "Healthy": "A lender should maintain the standard review cycle and watch for "
                   "debt-funded expansion rather than trading deterioration.",
        "Watch": "A lender should shorten the reporting cycle and re-test covenant headroom "
                 "before the next renewal.",
        "Elevated Risk": "A lender should request updated management accounts now and "
                         "reassess security cover rather than waiting for the next filing.",
        "Critical": "A lender should treat further drawdowns as refinancing, verify "
                    "collateral independently, and prepare for restructuring discussions.",
    }.get(f.combined_band, "A lender should keep this name under periodic review.")

    return " ".join([s1, s2, s3, s4])


# ---------------------------------------------------------------- narrative cache
# Pre-generated LLM narratives, committed to the repo so the demo works OFFLINE. The key
# is a hash of the exact prompt, so if a company's numbers change the cache misses and the
# text regenerates (or falls back) rather than showing stale prose. This is genuine LLM
# output, just persisted -- so a cache hit is still reported as 'llm'.
import hashlib
import json as _json
from pathlib import Path as _Path

_CACHE_PATH = _Path(__file__).resolve().parents[1] / "data" / "demo" / "narrative_cache.json"


def _cache_key(f: NarrativeFacts) -> str:
    return hashlib.sha256(_prompt(f).encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict:
    try:
        return _json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _store_cache(key: str, text: str) -> None:
    cache = _load_cache()
    cache[key] = text
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(_json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# -------------------------------------------------------------------- public API
def generate(
    combined: CombinedRisk,
    fin: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    sector: str = "",
    use_llm: bool = True,
    use_cache: bool = True,
    live: bool = True,
) -> tuple[str, str]:
    """Return `(narrative, source)` where source is 'llm' or 'rule-based'.

    Resolution order: pre-generated cache -> live LLM (and cache the result) -> rule-based.
    Never raises and never depends on the network when the cache is warm, which is what
    makes the live demo safe offline. With `live=False` the live LLM call is skipped
    entirely (cache -> rule-based), so the request path never blocks on the network -- used
    for the pre-warmed tracked roster, where a cache miss should degrade instantly.
    """
    facts = collect_facts(combined, fin, digital, sector)
    if use_llm:
        key = _cache_key(facts)
        if use_cache:
            cached = _load_cache().get(key)
            if cached:
                return cached, "llm"
        if live:
            text = _llm(facts)
            if text:
                _store_cache(key, text)
                return text, "llm"
    return _fallback(facts), "rule-based"


def prewarm_cache(quiet: bool = False) -> int:
    """Populate the narrative cache for the whole demo roster (run this online).

    Uses the same FinBERT-scored path the app uses, so the cache keys match exactly what
    the live dashboard will request. Returns the number of narratives cached.
    """

    scorer = default_scorer()  # FinBERT if available -- must match the app
    cache = _load_cache()
    n = 0
    for rec, prior, _ in ROSTER:
        fin = score_company(rec, prior=prior)
        dig = pulse_as_of(rec.company, DEFAULT_AS_OF[rec.company], scorer)
        sector = SECTORS.get(rec.company, "")  # MUST match the app's call
        combined = fuse(fin, dig)
        key = _cache_key(collect_facts(combined, fin, dig, sector))
        if key in cache:  # idempotent -- re-runs only fill gaps, sparing the rate limit
            n += 1
            if not quiet:
                print(f"  already cached: {rec.company}")
            continue
        text, src = generate(combined, fin, dig, sector=sector, use_cache=False)
        if src == "llm":
            n += 1
            if not quiet:
                print(f"  cached: {rec.company}")
        elif not quiet:
            print(f"  FALLBACK (not cached, retry): {rec.company}")
    return n

"""Risk Mitigation Recommendations.

**Rule-based by design, never LLM-generated** (rule-based by design). These sit next to a number
a credit committee may act on, so they must be reproducible and incapable of inventing a
fact. Every rule is tied to actual metric values, not generic advice.

Output is split by audience, which is what shows the platform serves multiple buyers:
  * FOR LENDERS / INVESTORS -- what to do if you have exposure.
  * FOR MANAGEMENT -- what to do if this is your own company.

Rules fire independently and are ranked by priority, so a company can trigger several.
"""


import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional



class Audience(str, Enum):
    LENDER = "For Lenders / Investors"
    MANAGEMENT = "For Management"


@dataclass(frozen=True)
class Recommendation:
    audience: Audience
    title: str
    action: str
    priority: int  # 1 = most urgent


@dataclass
class _Ctx:
    """Everything the rules read. Missing values are NaN/None and rules must tolerate it."""
    company: str
    fin: FinancialScore
    interest_cover: float
    #: total liabilities / assets -- INCLUDES trade payables and provisions, so this is
    #: not "debt". FMCG names run high here on supplier credit while carrying no borrowing.
    debt_to_assets: float
    #: interest-bearing borrowings / assets. Use THIS for any statement about leverage or
    #: debt, never `debt_to_assets`, or debt-free companies get told they are levered.
    borrowings_to_assets: float
    equity_ratio: float
    roa: float
    op_margin: float
    wc_to_assets: float
    digital: Optional[DigitalPulse]

    def sig(self, kind: SignalKind) -> Optional[float]:
        if self.digital is None:
            return None
        r = self.digital.by_kind(kind)
        return None if r is None else r.risk_score

    def sig_label(self, kind: SignalKind) -> str:
        if self.digital is None:
            return ""
        r = self.digital.by_kind(kind)
        return "" if r is None else r.label

    def sig_raw(self, kind: SignalKind) -> Optional[float]:
        if self.digital is None:
            return None
        r = self.digital.by_kind(kind)
        return None if r is None else r.raw


def _ok(x: Optional[float]) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


Rule = Callable[[_Ctx], Optional[Recommendation]]
_RULES: list[Rule] = []


def _rule(fn: Rule) -> Rule:
    _RULES.append(fn)
    return fn


# ------------------------------------------------------------------ lender rules
@_rule
def _r_negative_equity(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.equity_ratio) and c.equity_ratio < 0:
        return Recommendation(
            Audience.LENDER, "Negative net worth",
            f"Book equity is negative ({c.equity_ratio:.0%} of assets): liabilities exceed "
            "assets, so unsecured exposure has no equity cushion behind it. Escalate to "
            "special-mention, re-verify collateral values and enforce security where held.", 1)
    return None


@_rule
def _r_cannot_service_debt(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and c.interest_cover < 1.0:
        return Recommendation(
            Audience.LENDER, "Interest not covered by operations",
            f"Interest coverage is {c.interest_cover:.2f}x - operating profit does not cover "
            "the interest bill, so debt service is being funded from reserves or new "
            "borrowing. Treat further drawdown requests as refinancing, not growth.", 1)
    return None


@_rule
def _r_thin_cover_high_leverage(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and _ok(c.debt_to_assets) and 1.0 <= c.interest_cover < 2.0 and c.debt_to_assets > 0.65:
        return Recommendation(
            Audience.LENDER, "Debt servicing capacity is thin relative to leverage",
            f"Coverage of {c.interest_cover:.2f}x against liabilities at {c.debt_to_assets:.0%} "
            "of assets leaves little headroom for a rate or earnings shock. Reassess covenant "
            "levels and consider tightening reporting frequency to monthly.", 2)
    return None


@_rule
def _r_distress_zone(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.zone == "Distress":
        return Recommendation(
            Audience.LENDER, "Altman Z-Score in the distress zone",
            f"Z'' of {c.fin.z_score:.2f} sits below the 1.10 distress threshold. Move the "
            "name to the watchlist, refresh the internal rating, and require updated "
            "management accounts before any limit renewal.", 2)
    return None


@_rule
def _r_loss_making(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.roa) and c.roa < 0:
        return Recommendation(
            Audience.LENDER, "Loss-making at the asset level",
            f"Return on assets is {c.roa:.0%}: the asset base is destroying capital rather "
            "than generating it. Confirm whether losses are one-off or structural before "
            "extending the facility.", 2)
    return None


@_rule
def _r_negative_working_capital(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.wc_to_assets) and c.wc_to_assets < -0.10 and _ok(c.interest_cover) and c.interest_cover < 2.0:
        return Recommendation(
            Audience.LENDER, "Negative working capital alongside weak coverage",
            f"Working capital is {c.wc_to_assets:.0%} of assets while coverage is only "
            f"{c.interest_cover:.2f}x. Short-term obligations are being met from operating "
            "inflows with no buffer - a single collection delay could trigger default.", 2)
    return None


@_rule
def _r_leadership_exodus(c: _Ctx) -> Optional[Recommendation]:
    n = c.sig_raw(SignalKind.LEADERSHIP)
    if n is not None and n > 2:
        return Recommendation(
            Audience.LENDER, "Senior leadership turnover",
            f"{int(n)} senior departures filed in the last six months. Concentrated exits at "
            "board or KMP level often precede disclosure events; request a governance update "
            "and confirm who now holds financial authority.", 2)
    return None


@_rule
def _r_operational_stress(c: _Ctx) -> Optional[Recommendation]:
    hire = c.sig_label(SignalKind.HIRING)
    sent = c.sig(SignalKind.NEWS_SENTIMENT)
    if hire == "Contracting" and sent is not None and sent >= 50:
        return Recommendation(
            Audience.LENDER, "Workforce contraction with negative coverage",
            "Hiring is contracting while news sentiment is negative - a pattern that "
            "indicates operational stress not yet reflected in the reported financials. "
            "Request updated management accounts rather than waiting for the next filing.", 2)
    return None


@_rule
def _r_digital_worse_than_financial(c: _Ctx) -> Optional[Recommendation]:
    if c.digital is None:
        return None
    gap = c.digital.composite_score - c.fin.risk_score
    if gap > 15:
        return Recommendation(
            Audience.LENDER, "Market signals weaker than the financials suggest",
            f"Digital signals score {c.digital.composite_score:.0f} against a financial score "
            f"of {c.fin.risk_score:.0f}. Where market signals lead the statements, the next "
            "reporting period often confirms them - bring forward the review date.", 3)
    return None


@_rule
def _r_no_digital_coverage(c: _Ctx) -> Optional[Recommendation]:
    if c.digital is None:
        return Recommendation(
            Audience.LENDER, "No market-intelligence coverage",
            "This assessment rests on reported financials alone. Establish news, hiring and "
            "filing monitoring for this name so deterioration between reporting dates is "
            "visible.", 4)
    return None


@_rule
def _r_grey_zone_watch(c: _Ctx) -> Optional[Recommendation]:
    """The watchlist band is where a lender most needs guidance -- without this, a company
    that is neither clearly safe nor clearly distressed produces no lender action."""
    if c.fin.zone == "Grey" or c.fin.band == "Watch":
        lev = (f" with liabilities at {c.debt_to_assets:.0%} of assets"
               if _ok(c.debt_to_assets) and c.debt_to_assets > 0.6 else "")
        return Recommendation(
            Audience.LENDER, "Grey zone - neither clearly safe nor distressed",
            f"Z'' of {c.fin.z_score:.2f} falls between the distress and safe thresholds{lev}. "
            "This is the band where outcomes diverge most: move to semi-annual review, "
            "confirm covenant headroom, and track the market signals for early deterioration.", 3)
    return None


@_rule
def _r_leverage_concentration(c: _Ctx) -> Optional[Recommendation]:
    # Gated on actual borrowings: a company funded by supplier credit is not "levered".
    if _ok(c.borrowings_to_assets) and c.borrowings_to_assets >= 0.30:
        return Recommendation(
            Audience.LENDER, "High balance-sheet leverage",
            f"Interest-bearing borrowings fund {c.borrowings_to_assets:.0%} of the asset base, "
            "so a modest fall in asset values would erode the equity cushion. Re-test security "
            "cover against current, not historic, valuations.", 3)
    return None


@_rule
def _r_healthy_but_levered(c: _Ctx) -> Optional[Recommendation]:
    if (c.fin.band == "Healthy" and _ok(c.borrowings_to_assets) and c.borrowings_to_assets > 0.20
            and _ok(c.interest_cover) and c.interest_cover >= 2.0):
        return Recommendation(
            Audience.LENDER, "Sound today, but leverage is material",
            f"Borrowings are {c.borrowings_to_assets:.0%} of assets even though coverage is "
            f"comfortable at {c.interest_cover:.1f}x. Monitor covenant headroom under a rate "
            "rise; the profile is sound but not shock-proof.", 4)
    return None


@_rule
def _r_healthy_standard_cycle(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.band == "Healthy" and (c.digital is None or c.digital.band == "Healthy"):
        return Recommendation(
            Audience.LENDER, "No action indicated",
            f"Financial and market signals are both in the healthy band (Z'' "
            f"{c.fin.z_score:.1f}). Maintain the standard annual review cycle; no additional "
            "monitoring is warranted on current evidence.", 5)
    return None


# -------------------------------------------------------------- management rules
@_rule
def _r_mgmt_recapitalize(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.equity_ratio) and c.equity_ratio < 0:
        return Recommendation(
            Audience.MANAGEMENT, "Restore the equity base",
            "Net worth is negative, which constrains new borrowing and may breach covenants. "
            "Priority is a recapitalisation - rights issue, promoter infusion, or converting "
            "debt to equity - before refinancing terms deteriorate further.", 1)
    return None


@_rule
def _r_mgmt_refinance(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.interest_cover) and c.interest_cover < 1.5:
        return Recommendation(
            Audience.MANAGEMENT, "Reduce the debt service burden",
            f"At {c.interest_cover:.2f}x coverage, interest is consuming most or all of "
            "operating profit. Open refinancing discussions early - extending maturities and "
            "resetting covenants is far cheaper while the company is still performing.", 1)
    return None


@_rule
def _r_mgmt_working_capital(c: _Ctx) -> Optional[Recommendation]:
    # Negative working capital is a *strength* in FMCG and retail -- suppliers fund the
    # business. It is only a problem when the company is not otherwise healthy, so this
    # rule is gated on the band. Telling a debt-free FMCG name to "close the gap" would be
    # wrong advice and would cost credibility instantly.
    if c.fin.band == "Healthy":
        return None
    if _ok(c.wc_to_assets) and c.wc_to_assets < -0.10:
        return Recommendation(
            Audience.MANAGEMENT, "Close the working capital gap",
            f"Working capital is {c.wc_to_assets:.0%} of assets. Negotiate a committed "
            "working-capital facility and tighten receivables collection so operations are "
            "not dependent on continuous supplier financing.", 2)
    return None


@_rule
def _r_mgmt_margins(c: _Ctx) -> Optional[Recommendation]:
    if _ok(c.op_margin) and c.op_margin < 0.05:
        return Recommendation(
            Audience.MANAGEMENT, "Rebuild operating margin",
            f"Operating margin is {c.op_margin:.1%}, leaving no absorption for input-cost or "
            "demand shocks. A structural cost review will do more for the risk profile than "
            "further balance-sheet engineering.", 2)
    return None


@_rule
def _r_mgmt_leadership(c: _Ctx) -> Optional[Recommendation]:
    n = c.sig_raw(SignalKind.LEADERSHIP)
    if n is not None and n >= 2:
        return Recommendation(
            Audience.MANAGEMENT, "Stabilise the leadership team",
            f"{int(n)} senior departures in six months is visible to lenders and rating "
            "agencies. Publish a clear succession and retention plan; unexplained churn is "
            "read as a governance signal regardless of the underlying reason.", 2)
    return None


@_rule
def _r_mgmt_workforce(c: _Ctx) -> Optional[Recommendation]:
    if c.sig_label(SignalKind.HIRING) == "Contracting" or c.sig_label(SignalKind.EMPLOYEE) == "Declining":
        return Recommendation(
            Audience.MANAGEMENT, "Address workforce signals",
            "Hiring contraction and falling employee sentiment are externally visible and "
            "feed third-party risk models. Communicate the operating plan internally before "
            "attrition compounds the problem.", 3)
    return None


@_rule
def _r_mgmt_communication(c: _Ctx) -> Optional[Recommendation]:
    s = c.sig(SignalKind.NEWS_SENTIMENT)
    if s is not None and s >= 55:
        return Recommendation(
            Audience.MANAGEMENT, "Get ahead of the narrative",
            "Recent coverage is materially negative. Proactive disclosure on liquidity and "
            "the funding plan is more effective than silence - lenders price uncertainty "
            "more harshly than bad news.", 3)
    return None


@_rule
def _r_mgmt_maintain(c: _Ctx) -> Optional[Recommendation]:
    if c.fin.band == "Healthy":
        return Recommendation(
            Audience.MANAGEMENT, "Preserve the current position",
            f"The balance sheet is sound (Z'' {c.fin.z_score:.1f}). Maintain the current "
            "leverage discipline; the main risk to this profile is debt-funded expansion "
            "rather than trading performance.", 5)
    return None


# -------------------------------------------------------------------- public API
def build_context(
    rec: ScreenerFinancials,
    fin: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    prior: Optional[ScreenerFinancials] = None,
) -> _Ctx:
    f = compute_features(rec, prior=prior)
    bta = (rec.borrowings / rec.total_assets) if rec.total_assets else float("nan")
    return _Ctx(
        company=rec.company, fin=fin,
        interest_cover=f.get("Attr27", float("nan")),
        debt_to_assets=f.get("Attr2", float("nan")),
        borrowings_to_assets=bta,
        equity_ratio=f.get("Attr10", float("nan")),
        roa=f.get("Attr1", float("nan")),
        op_margin=f.get("Attr42", float("nan")),
        wc_to_assets=f.get("Attr3", float("nan")),
        digital=digital,
    )


def recommend(
    rec: ScreenerFinancials,
    fin: FinancialScore,
    digital: Optional[DigitalPulse] = None,
    prior: Optional[ScreenerFinancials] = None,
) -> dict[Audience, list[Recommendation]]:
    """Fire every rule and group the hits by audience, most urgent first."""
    ctx = build_context(rec, fin, digital, prior)
    out: dict[Audience, list[Recommendation]] = {Audience.LENDER: [], Audience.MANAGEMENT: []}
    for rule in _RULES:
        r = rule(ctx)
        if r is not None:
            out[r.audience].append(r)
    for aud in out:
        out[aud].sort(key=lambda r: r.priority)
    return out


def rule_count() -> int:
    return len(_RULES)

"""Executive Report: a two-page credit-intelligence memo (ReportLab).

Page 1  identity, combined risk score, key financial metrics, market signals, narrative.
Page 2  Altman decomposition, recommendations by audience, disclaimer.

Two deliberate choices on page 2, both evidence-driven:

* No SHAP waterfall. Our *serving* explanation is the exact Altman Z'' 4-term
  decomposition (the GBM does not drive the displayed score), so that is what we print.
* No debt/profitability *trajectory* charts. We curated a single fiscal year
  for seven of eight companies, so multi-year trajectories do not exist for them and are
  not invented. Where a prior year exists the year-on-year move is printed as a line of
  text instead.

The closing disclaimer is not weakness -- real enterprise risk reporting carries one.
"""


from datetime import datetime
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle


# Palette mirrors the dashboard so the report is recognisably the same product.
NAVY = colors.HexColor("#0A1628")
PANEL = colors.HexColor("#0F1E33")
AMBER = colors.HexColor("#F59E0B")
DIM = colors.HexColor("#94A3B8")
BORDER = colors.HexColor("#1E3350")
GOOD = colors.HexColor("#22C55E")
WATCH = colors.HexColor("#F59E0B")
ELEVATED = colors.HexColor("#F97316")
BAD = colors.HexColor("#EF4444")

_BAND_COLOR = {"Healthy": GOOD, "Watch": WATCH, "Elevated Risk": ELEVATED, "Critical": BAD}

_PAGE_W, _PAGE_H = A4
_MARGIN = 16 * mm
_BANNER_H = 22 * mm


def _band_color(band: str):
    return _BAND_COLOR.get(band, DIM)


# ------------------------------------------------------------------------ styles
def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9, leading=13,
                          textColor=colors.HexColor("#1F2937"), alignment=TA_LEFT)
    return {
        "body": base,
        "section": ParagraphStyle("section", parent=base, fontName="Helvetica-Bold",
                                  fontSize=8, textColor=colors.HexColor("#64748B"),
                                  spaceAfter=4, leading=11),
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=17,
                             textColor=NAVY, leading=20),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9, textColor=colors.HexColor("#64748B")),
        "narr": ParagraphStyle("narr", parent=base, fontSize=9.5, leading=14),
        "rec_t": ParagraphStyle("rec_t", parent=base, fontName="Helvetica-Bold", fontSize=8.5,
                                leading=11),
        "rec_b": ParagraphStyle("rec_b", parent=base, fontSize=8, leading=11,
                                textColor=colors.HexColor("#475569")),
        "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=DIM),
        "disc": ParagraphStyle("disc", parent=base, fontSize=7.5, leading=10,
                               textColor=colors.HexColor("#64748B")),
    }


# ------------------------------------------------------------- page furniture
def _draw_page(canvas, doc, company: str, generated: str) -> None:
    canvas.saveState()
    # Navy banner with the brand.
    canvas.setFillColor(NAVY)
    canvas.rect(0, _PAGE_H - _BANNER_H, _PAGE_W, _BANNER_H, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(_MARGIN, _PAGE_H - 14 * mm, "Foresight")
    canvas.setFillColor(AMBER)
    canvas.drawString(_MARGIN + 24 * mm, _PAGE_H - 14 * mm, "AI")
    canvas.setFillColor(DIM)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(_PAGE_W - _MARGIN, _PAGE_H - 14 * mm,
                           "CREDIT INTELLIGENCE MEMO")
    # Footer.
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(_MARGIN, 10 * mm, f"{company}  |  Generated {generated}")
    canvas.drawRightString(_PAGE_W - _MARGIN, 10 * mm, f"Page {canvas.getPageNumber()} of 2")
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.line(_MARGIN, 13 * mm, _PAGE_W - _MARGIN, 13 * mm)
    canvas.restoreState()


# ------------------------------------------------------------------- components
def _score_block(combined: CombinedRisk, fin: FinancialScore, st) -> Table:
    """Large coloured score box plus the two component legs."""
    c = _band_color(combined.band)
    score_cell = Table(
        [[Paragraph(f'<font size="30" color="white"><b>{combined.combined_score:.0f}</b></font>'
                    f'<font size="11" color="white">/100</font>', st["body"])],
         [Paragraph(f'<font size="10" color="white"><b>{combined.band.upper()}</b></font>', st["body"])]],
        colWidths=[46 * mm], rowHeights=[16 * mm, 7 * mm])
    score_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    dig = f"{combined.digital_score:.0f}/100" if combined.has_digital else "Not available"
    detail = Table([
        ["Financial Health", f"{combined.financial_score:.0f}/100",
         f"weight {combined.financial_weight:.0%}"],
        ["Market Signals", dig, f"weight {combined.digital_weight:.0%}"],
        ["Altman Z''", f"{fin.z_score:.2f}", f"{fin.zone} zone"],
    ], colWidths=[34 * mm, 26 * mm, 32 * mm])
    detail.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("FONT", (1, 0), (1, -1), "Helvetica-Bold", 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#94A3B8")),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    wrap = Table([[score_cell, detail]], colWidths=[50 * mm, 96 * mm])
    wrap.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (0, 0), 0)]))
    return wrap


def _metrics_table(rec: ScreenerFinancials, prior, st) -> Table:
    f = compute_features(rec, prior=prior)

    def fmt(key, pct=False, suffix=""):
        v = f.get(key, float("nan"))
        if v != v:
            return "n/a", DIM
        txt = f"{v*100:.0f}%" if pct else f"{v:.2f}{suffix}"
        return txt, None

    rows = [
        ("Interest Coverage", *fmt("Attr27", suffix="x"), lambda v: v >= 2, "Attr27"),
        ("Debt to Assets", *fmt("Attr2", pct=True), lambda v: v < 0.7, "Attr2"),
        ("Return on Assets", *fmt("Attr1", pct=True), lambda v: v > 0, "Attr1"),
        ("Equity Ratio", *fmt("Attr10", pct=True), lambda v: v > 0.2, "Attr10"),
    ]
    data, styles = [[], []], []
    for i, (label, txt, _dim, ok, key) in enumerate(rows):
        v = f.get(key, float("nan"))
        col = DIM if v != v else (GOOD if ok(v) else BAD)
        data[0].append(Paragraph(f'<font size="7" color="#64748B">{label.upper()}</font>', st["body"]))
        data[1].append(Paragraph(f'<font size="13" color="{col.hexval()}"><b>{txt}</b></font>', st["body"]))
    t = Table(data, colWidths=[36.5 * mm] * 4)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _signals_block(digital: Optional[DigitalPulse], st) -> list:
    if digital is None:
        return [Paragraph("Market-intelligence signals are not available for this company; "
                          "the assessment rests on reported financials alone.", st["body"])]
    out = []
    for r in digital.readings:
        col = _band_color(r.band)
        out.append(Paragraph(
            f'<font color="{col.hexval()}">•</font> <b>{r.kind.value}</b> '
            f'<font color="#64748B">({r.label})</font> - {r.datum}', st["body"]))
        out.append(Spacer(1, 2.5 * mm))
    return out


def _decomposition(fin: FinancialScore, st) -> Table:
    """Exact Altman Z'' 4-term decomposition -- the terms sum to the score."""
    maxc = max((abs(t.contribution) for t in fin.terms), default=1.0) or 1.0
    data = []
    for t in fin.terms:
        col = GOOD if t.contribution > 0 else BAD
        width = max(1.0, abs(t.contribution) / maxc * 42)
        bar = Table([[""]], colWidths=[width * mm], rowHeights=[3.4 * mm])
        bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        data.append([
            Paragraph(f'<font size="8">{t.label}</font>', st["body"]), bar,
            Paragraph(f'<font size="8" color="{col.hexval()}"><b>{t.contribution:+.2f}</b></font>'
                      f'<font size="7" color="#94A3B8">  ({t.value:+.2f} &times; {t.coefficient})</font>',
                      st["body"]),
        ])
    data.append([Paragraph('<font size="8"><b>Altman Z'' (sum)</b></font>', st["body"]), "",
                 Paragraph(f'<font size="8"><b>{fin.z_score:+.2f}</b></font>', st["body"])])
    t = Table(data, colWidths=[62 * mm, 44 * mm, 40 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    return t


def _recs_block(recs: dict[Audience, list[Recommendation]], st) -> list:
    out = []
    for aud in (Audience.LENDER, Audience.MANAGEMENT):
        items = recs.get(aud, [])
        out.append(Paragraph(f'<font color="#B45309"><b>{aud.value.upper()}</b></font>', st["section"]))
        out.append(Spacer(1, 1.5 * mm))
        if not items:
            out.append(Paragraph("No specific actions indicated.", st["rec_b"]))
        for it in items[:4]:  # keep the memo to two pages
            out.append(Paragraph(f"{it.title}", st["rec_t"]))
            out.append(Paragraph(it.action, st["rec_b"]))
            out.append(Spacer(1, 2 * mm))
        out.append(Spacer(1, 3 * mm))
    return out


def _yoy_line(rec: ScreenerFinancials, prior: Optional[ScreenerFinancials], st):
    """Year-on-year move where a prior year exists; otherwise say so plainly."""
    if prior is None:
        return Paragraph("Only a single reported year was available for this company, so a "
                         "multi-year trajectory is not shown.", st["small"])

    def move(now, then):
        """Percentage change, but fall back to absolute values when a percentage would
        mislead. Jet's operating profit went +24 -> -3,660: '-15350%' is arithmetically
        correct and completely meaningless to a reader, so print the figures instead."""
        if not then:
            return f"{then:,.0f} to {now:,.0f}"
        if (now < 0) != (then < 0):          # sign flip
            return f"{then:,.0f} to {now:,.0f}"
        change = (now - then) / abs(then) * 100
        if abs(change) > 300:                 # tiny base -> runaway percentage
            return f"{then:,.0f} to {now:,.0f}"
        return f"{change:+.0f}%"

    return Paragraph(
        f"Year on year ({prior.year} to {rec.year}): sales {move(rec.sales, prior.sales)}, "
        f"operating profit {move(rec.operating_profit, prior.operating_profit)}, "
        f"borrowings {move(rec.borrowings, prior.borrowings)}, "
        f"reserves {move(rec.reserves, prior.reserves)}. Figures in Rs crore.", st["small"])


# ------------------------------------------------------------------ public API
def build_report(
    rec: ScreenerFinancials,
    combined: CombinedRisk,
    fin: FinancialScore,
    digital: Optional[DigitalPulse],
    narrative: str,
    recommendations: dict[Audience, list[Recommendation]],
    sector: str = "",
    prior: Optional[ScreenerFinancials] = None,
) -> bytes:
    """Render the two-page memo and return the PDF bytes."""
    st = _styles()
    generated = datetime.now().strftime("%d %b %Y %H:%M")
    buf = BytesIO()

    doc = BaseDocTemplate(
        buf, pagesize=A4, leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_BANNER_H + 8 * mm, bottomMargin=18 * mm,
        title=f"Foresight AI - {rec.company}", author="Foresight AI Analytics Engine",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(
        id="main", frames=[frame],
        onPage=lambda c, d: _draw_page(c, d, rec.company, generated))])

    story: list = []
    # --- page 1 ---------------------------------------------------------
    story.append(Paragraph(rec.company, st["h1"]))
    meta = f"{sector} &middot; " if sector else ""
    story.append(Paragraph(f"{meta}FY{rec.year} reported financials &middot; "
                           f"Analyst: Foresight AI Analytics Engine", st["sub"]))
    story.append(Spacer(1, 5 * mm))
    story.append(_score_block(combined, fin, st))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("KEY FINANCIAL METRICS", st["section"]))
    story.append(_metrics_table(rec, prior, st))
    story.append(Spacer(1, 2.5 * mm))
    story.append(_yoy_line(rec, prior, st))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("MARKET INTELLIGENCE SIGNALS", st["section"]))
    story.extend(_signals_block(digital, st))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("ANALYST SUMMARY", st["section"]))
    story.append(Paragraph(narrative, st["narr"]))

    # --- page 2 ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("WHY THIS SCORE - ALTMAN Z'' DECOMPOSITION", st["section"]))
    story.append(Spacer(1, 1.5 * mm))
    story.append(_decomposition(fin, st))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("The four terms are exact and sum to the Z'' score; green "
                           "reduces risk, red increases it.", st["small"]))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("RECOMMENDED ACTIONS", st["section"]))
    story.append(Spacer(1, 1.5 * mm))
    story.extend(_recs_block(recommendations, st))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "This report is generated by an AI analytics system and should be used as a "
        "supplementary tool alongside professional financial analysis. Financial scores "
        "are derived from reported financial statements using the Altman Z'' model. "
        "Market-intelligence signals may include illustrative data and are labelled where "
        "so; they are not a substitute for verified disclosure.", st["disc"]))

    doc.build(story)
    return buf.getvalue()

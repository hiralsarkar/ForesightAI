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

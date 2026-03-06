from pathlib import Path
import numpy as np
import pandas as pd


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize a pandas Series at the given lower and upper quantiles.
    """
    lower_q = series.quantile(lower)
    upper_q = series.quantile(upper)
    return series.clip(lower=lower_q, upper=upper_q)


def clean_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Ensure selected columns are numeric.
    Non-numeric values are coerced to NaN.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_compustat(raw_path: str, clean_path: str, config: dict) -> pd.DataFrame:
    """
    Build a cleaned Compustat annual panel for empirical corporate finance work.

    Current cleaning steps:
    1. Load raw WRDS Compustat annual data
    2. Standardize numeric columns
    3. Drop duplicate firm-years
    4. Remove financial firms (SIC 6000-6999)
    5. Remove utilities (SIC 4900-4999)
    6. Require non-missing core accounting variables
    7. Require positive assets and positive capital
    8. Construct lagged capital and core empirical variables
    9. Remove extreme accounting/pathological observations
    10. Winsorize key ratios
    11. Sort panel and save cleaned file
    """

    raw_p = Path(raw_path)
    clean_p = Path(clean_path)
    clean_p.parent.mkdir(parents=True, exist_ok=True)

    if not raw_p.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_p}")

    print("Loading raw Compustat data...")
    df = pd.read_csv(raw_p)
    print(f"Initial rows: {len(df):,}")

    # -----------------------------
    # Keep only the columns we need
    # -----------------------------
    required_cols = [
        "gvkey",
        "fyear",
        "sic",
        "at",
        "ppent",
        "capx",
        "dltt",
        "dlc",
        "che",
        "oibdp",
        "sale",
        "dp",
        "ib",
        "lt",
        "prcc_f",
        "csho",
    ]

    existing_cols = [col for col in required_cols if col in df.columns]
    df = df[existing_cols].copy()

    # -----------------------------
    # Standardize numeric columns
    # -----------------------------
    numeric_cols = [
        "fyear",
        "sic",
        "at",
        "ppent",
        "capx",
        "dltt",
        "dlc",
        "che",
        "oibdp",
        "sale",
        "dp",
        "ib",
        "lt",
        "prcc_f",
        "csho",
    ]
    df = clean_numeric_columns(df, numeric_cols)

    # -----------------------------
    # Drop duplicate firm-years
    # -----------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["gvkey", "fyear"])
    print(f"After dropping duplicate firm-years: {len(df):,} (removed {before - len(df):,})")

    # -----------------------------
    # Remove financial firms
    # SIC 6000-6999
    # -----------------------------
    before = len(df)
    df = df[~df["sic"].between(6000, 6999, inclusive="both")]
    print(f"After removing financials: {len(df):,} (removed {before - len(df):,})")

    # -----------------------------
    # Remove utilities
    # SIC 4900-4999
    # -----------------------------
    before = len(df)
    df = df[~df["sic"].between(4900, 4999, inclusive="both")]
    print(f"After removing utilities: {len(df):,} (removed {before - len(df):,})")

    # -----------------------------
    # Drop missing core variables
    # -----------------------------
    core_vars = ["gvkey", "fyear", "sic", "at", "ppent", "capx", "dltt", "dlc", "che", "oibdp", "sale", "dp"]
    before = len(df)
    df = df.dropna(subset=core_vars)
    print(f"After dropping missing core vars: {len(df):,} (removed {before - len(df):,})")

    # -----------------------------
    # Basic economic restrictions
    # -----------------------------
    before = len(df)
    df = df[df["at"] > 0]
    print(f"After requiring positive assets: {len(df):,} (removed {before - len(df):,})")

    before = len(df)
    df = df[df["ppent"] > 0]
    print(f"After requiring positive capital: {len(df):,} (removed {before - len(df):,})")

    # -----------------------------
    # Sort panel before lagging
    # -----------------------------
    df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    # -----------------------------
    # Construct core variables
    # -----------------------------
    df["capital"] = df["ppent"]
    df["debt"] = df["dltt"] + df["dlc"]

    # Lagged capital by firm
    df["capital_lag"] = df.groupby("gvkey")["capital"].shift(1)

    before = len(df)
    df = df.dropna(subset=["capital_lag"])
    print(f"After requiring lagged capital: {len(df):,} (removed {before - len(df):,})")

    # Guard against zero/negative lagged capital
    before = len(df)
    df = df[df["capital_lag"] > 0]
    print(f"After requiring positive lagged capital: {len(df):,} (removed {before - len(df):,})")

    # -----------------------------
    # Main empirical ratios
    # -----------------------------
    # Canonical definitions for the project:
    # - investment     = capx / capital_lag
    # - leverage       = debt / at
    # - cash_ratio     = che / at
    # - profitability  = oibdp / capital_lag
    # These definitions should match the report and the simulated-moment targets.
    df["investment"] = df["capx"] / df["capital_lag"]
    df["leverage"] = df["debt"] / df["at"]
    df["cash_ratio"] = df["che"] / df["at"]
    df["profitability"] = df["oibdp"] / df["capital_lag"]
    df["sales_to_assets"] = df["sale"] / df["at"]
    df["depreciation_rate"] = df["dp"] / df["capital_lag"]

    # Optional variables for later robustness work
    if "prcc_f" in df.columns and "csho" in df.columns:
        df["market_equity"] = df["prcc_f"] * df["csho"]

    if "lt" in df.columns:
        df["liabilities_to_assets"] = df["lt"] / df["at"]

    if "ib" in df.columns:
        df["income_to_capital"] = df["ib"] / df["capital_lag"]

    # -----------------------------
    # Remove pathological values
    # -----------------------------
    before = len(df)
    df = df[np.isfinite(df["investment"])]
    df = df[np.isfinite(df["leverage"])]
    df = df[np.isfinite(df["cash_ratio"])]
    df = df[np.isfinite(df["profitability"])]
    df = df[np.isfinite(df["sales_to_assets"])]
    df = df[np.isfinite(df["depreciation_rate"])]
    print(f"After removing non-finite ratios: {len(df):,} (removed {before - len(df):,})")

    # Mild research-standard trimming before winsorization
    before = len(df)
    df = df[(df["investment"] > -1.0) & (df["investment"] < 2.0)]
    df = df[(df["leverage"] >= 0.0) & (df["leverage"] < 5.0)]
    df = df[(df["cash_ratio"] >= 0.0) & (df["cash_ratio"] < 5.0)]
    print(f"After trimming extreme ratios: {len(df):,} (removed {before - len(df):,})")

    # -----------------------------
    # Winsorize main empirical ratios
    # -----------------------------
    winsor_cols = [
        "investment",
        "leverage",
        "cash_ratio",
        "profitability",
        "sales_to_assets",
        "depreciation_rate",
    ]

    for col in winsor_cols:
        df[col] = winsorize_series(df[col], lower=0.01, upper=0.99)

    # -----------------------------
    # Final sort
    # -----------------------------
    df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    print(f"Final cleaned rows: {len(df):,}")

    # -----------------------------
    # Save cleaned file
    # -----------------------------
    df.to_csv(clean_p, index=False)
    print(f"Clean dataset saved to: {clean_p}")

    return df


def load_clean_data(clean_path: str) -> pd.DataFrame:
    """
    Load the cleaned Compustat dataset.
    """
    p = Path(clean_path)

    if not p.exists():
        raise FileNotFoundError(f"Clean data not found at: {p}")

    return pd.read_csv(p)


def make_subsamples(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """
    Placeholder subsamples for later estimation.
    For now:
    - full sample
    - small firms
    - large firms

    Size split is based on median total assets.
    """
    median_assets = df["at"].median()

    subsamples = {
        "full": df.copy(),
        "small": df[df["at"] <= median_assets].copy(),
        "large": df[df["at"] > median_assets].copy(),
    }

    return subsamples
import numpy as np
import pandas as pd


def moment_names(config: dict) -> list[str]:
    """
    Names of empirical moments used in SMM estimation.

    Because we plan to implement the cash extension, we target
    mean / variance / autocorrelation blocks for the core model objects:
    investment, leverage, profitability, and cash.
    """
    return [
        "mean_investment",
        "var_investment",
        "autocorr_investment",
        "mean_leverage",
        "var_leverage",
        "autocorr_leverage",
        "mean_profitability",
        "var_profitability",
        "autocorr_profitability",
        "mean_cash_ratio",
        "var_cash_ratio",
        "autocorr_cash_ratio",
    ]


def _pooled_within_firm_autocorr(
    df: pd.DataFrame,
    firm_col: str,
    time_col: str,
    value_col: str,
) -> float:
    """
    Compute pooled within-firm first-order autocorrelation.

    Procedure:
    1. Sort by firm and time
    2. Create firm-specific lag
    3. Correlate current and lagged values across all valid firm-year observations
    """
    work = df[[firm_col, time_col, value_col]].copy()
    work = work.sort_values([firm_col, time_col]).reset_index(drop=True)
    work[f"{value_col}_lag"] = work.groupby(firm_col)[value_col].shift(1)

    valid = work.dropna(subset=[value_col, f"{value_col}_lag"])
    if len(valid) < 2:
        return np.nan

    return valid[value_col].corr(valid[f"{value_col}_lag"])


def compute_moments(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Compute empirical moments from the cleaned Compustat panel.

    Required columns:
    - gvkey
    - fyear
    - investment
    - leverage
    - profitability
    - cash_ratio
    """
    required_cols = [
        "gvkey",
        "fyear",
        "investment",
        "leverage",
        "profitability",
        "cash_ratio",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for moment computation: {missing}")

    work = df[required_cols].copy()
    work = work.sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    # -----------------------------
    # Investment moments
    # -----------------------------
    mean_investment = work["investment"].mean()
    var_investment = work["investment"].var(ddof=1)
    autocorr_investment = _pooled_within_firm_autocorr(
        df=work,
        firm_col="gvkey",
        time_col="fyear",
        value_col="investment",
    )

    # -----------------------------
    # Leverage moments
    # -----------------------------
    mean_leverage = work["leverage"].mean()
    var_leverage = work["leverage"].var(ddof=1)
    autocorr_leverage = _pooled_within_firm_autocorr(
        df=work,
        firm_col="gvkey",
        time_col="fyear",
        value_col="leverage",
    )

    # -----------------------------
    # Profitability moments
    # -----------------------------
    mean_profitability = work["profitability"].mean()
    var_profitability = work["profitability"].var(ddof=1)
    autocorr_profitability = _pooled_within_firm_autocorr(
        df=work,
        firm_col="gvkey",
        time_col="fyear",
        value_col="profitability",
    )

    # -----------------------------
    # Cash moments
    # -----------------------------
    mean_cash_ratio = work["cash_ratio"].mean()
    var_cash_ratio = work["cash_ratio"].var(ddof=1)
    autocorr_cash_ratio = _pooled_within_firm_autocorr(
        df=work,
        firm_col="gvkey",
        time_col="fyear",
        value_col="cash_ratio",
    )

    moments = np.array(
        [
            mean_investment,
            var_investment,
            autocorr_investment,
            mean_leverage,
            var_leverage,
            autocorr_leverage,
            mean_profitability,
            var_profitability,
            autocorr_profitability,
            mean_cash_ratio,
            var_cash_ratio,
            autocorr_cash_ratio,
        ],
        dtype=float,
    )

    return moments
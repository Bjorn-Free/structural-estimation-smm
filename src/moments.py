import numpy as np
import pandas as pd


def moment_names(config: dict) -> list[str]:
    """
    Names of empirical moments used in SMM estimation.

    After dropping leverage from the structural target set, we match
    mean / variance / autocorrelation blocks for:
    - investment
    - profitability
    - cash_ratio
    """
    return [
        "mean_investment",
        "var_investment",
        "autocorr_investment",
        "mean_profitability",
        "var_profitability",
        "autocorr_profitability",
        "mean_cash_ratio",
        "var_cash_ratio",
        "autocorr_cash_ratio",
    ]


def _validate_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the dataframe contains all required columns for moment work.
    """
    required_cols = [
        "gvkey",
        "fyear",
        "investment",
        "profitability",
        "cash_ratio",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for moment computation: {missing}")

    work = df[required_cols].copy()
    work = work.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    return work


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

    return float(valid[value_col].corr(valid[f"{value_col}_lag"]))


def _safe_mean(series: pd.Series) -> float:
    """
    Compute a mean safely.
    """
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def _safe_variance(series: pd.Series) -> float:
    """
    Compute a sample variance safely.
    """
    if len(series) < 2:
        return np.nan
    return float(series.var(ddof=1))


def _autocorr_components(
    df: pd.DataFrame,
    firm_col: str,
    time_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Construct the components needed to compute a pooled within-firm
    first-order autocorrelation and its moment contribution.

    Returns a dataframe with valid current/lag observations only.
    """
    work = df[[firm_col, time_col, value_col]].copy()
    work = work.sort_values([firm_col, time_col]).reset_index(drop=True)
    lag_col = f"{value_col}_lag"
    work[lag_col] = work.groupby(firm_col)[value_col].shift(1)

    valid = work.dropna(subset=[value_col, lag_col]).copy()
    if valid.empty:
        return valid

    x = valid[value_col].astype(float)
    y = valid[lag_col].astype(float)

    mx = float(x.mean())
    my = float(y.mean())

    x_dev = x - mx
    y_dev = y - my

    valid["x_dev"] = x_dev
    valid["y_dev"] = y_dev
    valid["cross"] = x_dev * y_dev
    valid["x_sq"] = x_dev**2
    valid["y_sq"] = y_dev**2

    return valid


def compute_moments(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Compute empirical moments from the cleaned Compustat panel.

    Required columns:
    - gvkey
    - fyear
    - investment
    - profitability
    - cash_ratio
    """
    work = _validate_required_columns(df)

    # -----------------------------
    # Investment moments
    # -----------------------------
    mean_investment = _safe_mean(work["investment"])
    var_investment = _safe_variance(work["investment"])
    autocorr_investment = _pooled_within_firm_autocorr(
        df=work,
        firm_col="gvkey",
        time_col="fyear",
        value_col="investment",
    )

    # -----------------------------
    # Profitability moments
    # -----------------------------
    mean_profitability = _safe_mean(work["profitability"])
    var_profitability = _safe_variance(work["profitability"])
    autocorr_profitability = _pooled_within_firm_autocorr(
        df=work,
        firm_col="gvkey",
        time_col="fyear",
        value_col="profitability",
    )

    # -----------------------------
    # Cash moments
    # -----------------------------
    mean_cash_ratio = _safe_mean(work["cash_ratio"])
    var_cash_ratio = _safe_variance(work["cash_ratio"])
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


def moment_contributions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Construct firm-level moment contributions for the efficient SMM weighting matrix.

    The output has one row per firm (gvkey) and one column per model moment.
    Taking the covariance matrix of these firm-level contributions gives a
    cluster-robust covariance estimator for the moment vector.

    Notes
    -----
    - Mean moments use firm-average contributions of x.
    - Variance moments use firm-average contributions of (x - mean_x)^2.
    - Autocorrelation moments use firm-average contributions of the three
      underlying pooled objects:
          E[(x_t - mx)(x_t-1 - my)]
          E[(x_t - mx)^2]
          E[(x_t-1 - my)^2]
      and then apply:
          corr = cov_xy / sqrt(var_x * var_y)

    This is a practical influence-function-style approximation suitable for
    efficient SMM weighting in the current project.
    """
    work = _validate_required_columns(df)
    firms = pd.DataFrame({"gvkey": np.sort(work["gvkey"].unique())})

    output = firms.copy()

    variables = [
        "investment",
        "profitability",
        "cash_ratio",
    ]

    for var in variables:
        series = work[var].astype(float)

        # -----------------------------
        # Mean contribution
        # -----------------------------
        mean_by_firm = work.groupby("gvkey")[var].mean().rename(f"mean_{var}")

        # -----------------------------
        # Variance contribution
        # -----------------------------
        grand_mean = float(series.mean())
        var_term = ((series - grand_mean) ** 2).astype(float)
        var_by_firm = (
            pd.DataFrame({"gvkey": work["gvkey"], f"var_{var}": var_term})
            .groupby("gvkey")[f"var_{var}"]
            .mean()
        )

        output = output.merge(mean_by_firm, on="gvkey", how="left")
        output = output.merge(var_by_firm, on="gvkey", how="left")

        # -----------------------------
        # Autocorrelation contribution
        # -----------------------------
        ac = _autocorr_components(
            df=work,
            firm_col="gvkey",
            time_col="fyear",
            value_col=var,
        )

        if ac.empty:
            output[f"autocorr_{var}"] = 0.0
            continue

        cov_xy = float(ac["cross"].mean())
        var_x = float(ac["x_sq"].mean())
        var_y = float(ac["y_sq"].mean())

        denom = np.sqrt(max(var_x, 1e-12) * max(var_y, 1e-12))
        rho = cov_xy / denom

        # Delta-method-style contribution for correlation
        ac["autocorr_contrib"] = (
            (ac["cross"] - cov_xy) / denom
            - 0.5 * rho * (ac["x_sq"] - var_x) / max(var_x, 1e-12)
            - 0.5 * rho * (ac["y_sq"] - var_y) / max(var_y, 1e-12)
        )

        ac_by_firm = (
            ac.groupby("gvkey")["autocorr_contrib"]
            .mean()
            .rename(f"autocorr_{var}")
        )
        output = output.merge(ac_by_firm, on="gvkey", how="left")

    ordered_cols = ["gvkey"] + moment_names(config)
    output = output[ordered_cols].copy()

    # Firms with no usable contribution for a moment get zero contribution.
    # This is especially relevant for autocorrelation moments when a firm
    # lacks enough time-series observations after lagging.
    output[moment_names(config)] = output[moment_names(config)].fillna(0.0)

    if output[moment_names(config)].isna().any().any():
        raise ValueError("Moment contributions still contain NaN values after fill.")

    return output


def moment_covariance_matrix(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Estimate the covariance matrix of the empirical sample moment vector
    using firm-level clustered moment contributions.

    Important:
    The covariance of the sample mean moment vector equals the covariance
    across firm-level contributions divided by the number of firms.

    This scaling is the correct object for:
        W = Var(m_data)^(-1)
    and for the SMM sandwich formula for standard errors.
    """
    contrib = moment_contributions(df, config)
    M = contrib[moment_names(config)].to_numpy(dtype=float)

    n_firms = M.shape[0]
    if n_firms < 2:
        raise ValueError("Need at least two firms to estimate moment covariance.")

    S_firm = np.cov(M, rowvar=False, ddof=1)
    S = S_firm / n_firms

    n_moments = int(config["n_moments"])
    if S.shape != (n_moments, n_moments):
        raise ValueError("Moment covariance matrix has incorrect dimensions.")

    return S
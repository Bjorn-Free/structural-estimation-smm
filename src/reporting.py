from pathlib import Path

import numpy as np
import pandas as pd


TABLE_DIR = Path("results/tables")


def _ensure_table_dir():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _format_numeric_table(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def _save_table(df: pd.DataFrame, filename_stem: str, decimals: int = 4):
    _ensure_table_dir()

    df_pretty = _format_numeric_table(df, decimals=decimals)

    csv_path = TABLE_DIR / f"{filename_stem}.csv"
    tex_path = TABLE_DIR / f"{filename_stem}.tex"

    df_pretty.to_csv(csv_path, index=False)

    latex_str = df_pretty.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.{decimals}f}",
    )
    tex_path.write_text(latex_str, encoding="utf-8")

    return df_pretty, csv_path, tex_path


def make_summary_statistics_table(df: pd.DataFrame):
    """
    Descriptive table for the cleaned Compustat sample.

    Keeping leverage here is fine as a descriptive variable even though it is
    no longer part of the targeted SMM moment vector.
    """
    variables = [
        ("Investment / Capital", "investment"),
        ("Debt / Assets", "leverage"),
        ("Operating Profit / Capital", "profitability"),
        ("Cash / Assets", "cash_ratio"),
        ("Sales / Assets", "sales_to_assets"),
        ("Depreciation / Capital", "depreciation_rate"),
    ]

    rows = []

    for label, col in variables:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        rows.append(
            {
                "variable": label,
                "n": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
            }
        )

    return pd.DataFrame(rows)


def save_summary_statistics_table(df: pd.DataFrame, decimals: int = 4):
    summary_df = make_summary_statistics_table(df)
    return _save_table(summary_df, "summary_statistics", decimals=decimals)


def make_moment_comparison_table(moment_labels, m_data, m_sim):
    label_map = {
        "mean_investment": "Mean investment",
        "var_investment": "Variance investment",
        "autocorr_investment": "Autocorrelation investment",
        "mean_profitability": "Mean profitability",
        "var_profitability": "Variance profitability",
        "autocorr_profitability": "Autocorrelation profitability",
        "mean_cash_ratio": "Mean cash ratio",
        "var_cash_ratio": "Variance cash ratio",
        "autocorr_cash_ratio": "Autocorrelation cash ratio",
    }

    pretty_labels = [label_map.get(x, x) for x in moment_labels]

    df = pd.DataFrame(
        {
            "Moment": pretty_labels,
            "Data": np.asarray(m_data, dtype=float),
            "Simulated": np.asarray(m_sim, dtype=float),
        }
    )
    df["Gap"] = df["Simulated"] - df["Data"]
    df["Absolute gap"] = df["Gap"].abs()
    return df


def save_moment_comparison_table(moment_labels, m_data, m_sim, decimals: int = 4):
    df = make_moment_comparison_table(moment_labels, m_data, m_sim)
    return _save_table(df, "moment_comparison", decimals=decimals)


def make_parameter_table(theta_hat, param_names=None, std_errors=None):
    theta_hat = np.asarray(theta_hat, dtype=float)

    if param_names is None:
        param_names = [f"theta_{i}" for i in range(len(theta_hat))]

    param_map = {
        "psi_adjustment_cost": r"$\psi$ adjustment cost",
        "lambda_external_finance_cost": r"$\lambda$ external finance cost",
    }
    pretty_names = [param_map.get(x, x) for x in param_names]

    df = pd.DataFrame(
        {
            "parameter": pretty_names,
            "estimate": theta_hat,
        }
    )

    if std_errors is not None:
        df["std_error"] = np.asarray(std_errors, dtype=float)

    return df


def save_parameter_table(theta_hat, param_names=None, std_errors=None, decimals: int = 4):
    df = make_parameter_table(theta_hat, param_names=param_names, std_errors=std_errors)
    return _save_table(df, "parameter_estimates", decimals=decimals)


def make_estimation_settings_table(config, objective_value, weighting_matrix_name="Identity"):
    sim_cfg = config.get("simulation", {})

    rows = [
        {"setting": "Weighting matrix", "value": weighting_matrix_name},
        {"setting": "Objective value", "value": float(objective_value)},
        {"setting": "Rebuild data", "value": config.get("rebuild_data", False)},
        {"setting": "Number of moments", "value": config.get("n_moments")},
        {"setting": "Initial parameter guess", "value": str(config.get("theta0"))},
        {"setting": "Bounds", "value": str(config.get("bounds"))},
        {"setting": "Number of firms", "value": sim_cfg.get("n_firms")},
        {"setting": "Simulation periods", "value": sim_cfg.get("t_periods")},
        {"setting": "Burn-in periods", "value": sim_cfg.get("burn_in")},
        {"setting": "Random seed", "value": sim_cfg.get("seed")},
    ]

    return pd.DataFrame(rows)


def save_estimation_settings_table(
    config,
    objective_value,
    weighting_matrix_name="Identity",
    decimals: int = 4,
):
    df = make_estimation_settings_table(
        config=config,
        objective_value=objective_value,
        weighting_matrix_name=weighting_matrix_name,
    )
    return _save_table(df, "estimation_settings", decimals=decimals)
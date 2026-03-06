from pathlib import Path

import numpy as np
import pandas as pd


TABLE_DIR = Path("results/tables")


def _ensure_table_dir():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _format_numeric_table(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    Return a copy of the table with numeric columns rounded for presentation.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def _save_table(df: pd.DataFrame, filename_stem: str, decimals: int = 4):
    """
    Save table as both CSV and LaTeX.
    """
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
    Build summary statistics table for the cleaned Compustat sample.
    """
    variables = {
        "investment": "investment",
        "leverage": "leverage",
        "profitability": "profitability",
        "cash_ratio": "cash_ratio",
        "sales_to_assets": "sales_to_assets",
        "depreciation_rate": "depreciation_rate",
    }

    rows = []

    for label, col in variables.items():
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
    df = pd.DataFrame(
        {
            "moment": moment_labels,
            "data": np.asarray(m_data, dtype=float),
            "simulated": np.asarray(m_sim, dtype=float),
        }
    )
    df["gap"] = df["simulated"] - df["data"]
    df["abs_gap"] = df["gap"].abs()
    return df


def save_moment_comparison_table(moment_labels, m_data, m_sim, decimals: int = 4):
    df = make_moment_comparison_table(moment_labels, m_data, m_sim)
    return _save_table(df, "moment_comparison", decimals=decimals)


def make_parameter_table(theta_hat, param_names=None, std_errors=None):
    theta_hat = np.asarray(theta_hat, dtype=float)

    if param_names is None:
        param_names = [f"theta_{i}" for i in range(len(theta_hat))]

    df = pd.DataFrame(
        {
            "parameter": param_names,
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
        {"setting": "weighting_matrix", "value": weighting_matrix_name},
        {"setting": "objective_value", "value": float(objective_value)},
        {"setting": "rebuild_data", "value": config.get("rebuild_data", False)},
        {"setting": "n_moments", "value": config.get("n_moments")},
        {"setting": "theta0", "value": str(config.get("theta0"))},
        {"setting": "bounds", "value": str(config.get("bounds"))},
        {"setting": "n_firms", "value": sim_cfg.get("n_firms")},
        {"setting": "t_periods", "value": sim_cfg.get("t_periods")},
        {"setting": "burn_in", "value": sim_cfg.get("burn_in")},
        {"setting": "seed", "value": sim_cfg.get("seed")},
    ]

    return pd.DataFrame(rows)


def save_estimation_settings_table(config, objective_value, weighting_matrix_name="Identity", decimals: int = 4):
    df = make_estimation_settings_table(
        config=config,
        objective_value=objective_value,
        weighting_matrix_name=weighting_matrix_name,
    )
    return _save_table(df, "estimation_settings", decimals=decimals)
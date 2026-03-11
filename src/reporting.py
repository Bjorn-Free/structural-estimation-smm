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


def _pretty_moment_label(name: str) -> str:
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
    return label_map.get(name, name)


def _pretty_param_label(name: str) -> str:
    param_map = {
        "psi_adjustment_cost": r"$\psi$ adjustment cost",
        "lambda_external_finance_cost": r"$\lambda$ external finance cost",
        "rho_productivity_persistence": r"$\rho$ productivity persistence",
        "sigma_productivity_volatility": r"$\sigma$ productivity volatility",
        "psi": r"$\psi$ adjustment cost",
        "lambda_external_finance": r"$\lambda$ external finance cost",
        "rho": r"$\rho$ productivity persistence",
        "sigma": r"$\sigma$ productivity volatility",
    }
    return param_map.get(name, name)


def _display_value(value, decimals: int = 4):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return round(float(value), decimals)
        return value
    return str(value)


def make_summary_statistics_table(df: pd.DataFrame):
    """
    Descriptive table for the cleaned Compustat sample.
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
    pretty_labels = [_pretty_moment_label(x) for x in moment_labels]

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


def save_moment_comparison_table(
    moment_labels,
    m_data,
    m_sim,
    decimals: int = 4,
    filename_stem: str = "moment_comparison",
):
    df = make_moment_comparison_table(moment_labels, m_data, m_sim)
    return _save_table(df, filename_stem, decimals=decimals)


def make_parameter_table(theta_hat, param_names=None, std_errors=None):
    theta_hat = np.asarray(theta_hat, dtype=float)

    if param_names is None:
        param_names = [f"theta_{i}" for i in range(len(theta_hat))]

    pretty_names = [_pretty_param_label(x) for x in param_names]

    df = pd.DataFrame(
        {
            "parameter": pretty_names,
            "estimate": theta_hat,
        }
    )

    if std_errors is not None:
        std_errors = np.asarray(std_errors, dtype=float)
        df["std_error"] = std_errors
        with np.errstate(divide="ignore", invalid="ignore"):
            df["t_stat"] = theta_hat / std_errors

    return df


def save_parameter_table(
    theta_hat,
    param_names=None,
    std_errors=None,
    decimals: int = 4,
    filename_stem: str = "parameter_estimates",
):
    df = make_parameter_table(theta_hat, param_names=param_names, std_errors=std_errors)
    return _save_table(df, filename_stem, decimals=decimals)


def make_estimation_settings_table(
    config,
    objective_value,
    weighting_matrix_name="Identity",
    sample_label=None,
    n_obs=None,
    weighting_details=None,
    optimizer_method=None,
    optimizer_options=None,
):
    sim_cfg = config.get("simulation", {})

    rows = [
        {"setting": "Sample label", "value": _display_value(sample_label)},
        {"setting": "Objective value", "value": _display_value(objective_value)},
        {"setting": "Weighting matrix", "value": _display_value(weighting_matrix_name)},
        {"setting": "Rebuild data", "value": _display_value(config.get("rebuild_data", False))},
        {"setting": "Debug mode", "value": _display_value(config.get("debug_mode", False))},
        {"setting": "Validation mode", "value": _display_value(config.get("validation_mode", "fast"))},
        {"setting": "Number of observations", "value": _display_value(n_obs)},
        {"setting": "Number of moments", "value": _display_value(config.get("n_moments"))},
        {"setting": "Initial parameter guess", "value": _display_value(config.get("theta0"))},
        {"setting": "Bounds", "value": _display_value(config.get("bounds"))},
        {"setting": "Optimizer method", "value": _display_value(optimizer_method)},
        {"setting": "Optimizer options", "value": _display_value(optimizer_options)},
        {"setting": "Number of firms", "value": _display_value(sim_cfg.get("n_firms"))},
        {"setting": "Simulation periods", "value": _display_value(sim_cfg.get("t_periods"))},
        {"setting": "Burn-in periods", "value": _display_value(sim_cfg.get("burn_in"))},
        {"setting": "Random seed", "value": _display_value(sim_cfg.get("seed"))},
    ]

    if weighting_details is not None:
        rows.extend(
            [
                {
                    "setting": "Weighting ridge scale",
                    "value": _display_value(weighting_details.get("ridge_scale")),
                },
                {
                    "setting": "Weighting condition number",
                    "value": _display_value(weighting_details.get("condition_number")),
                },
                {
                    "setting": "Weighting used pseudo-inverse",
                    "value": _display_value(weighting_details.get("used_pinv")),
                },
            ]
        )

    return pd.DataFrame(rows)


def save_estimation_settings_table(
    config,
    objective_value,
    weighting_matrix_name="Identity",
    decimals: int = 4,
    filename_stem: str = "estimation_settings",
    sample_label=None,
    n_obs=None,
    weighting_details=None,
    optimizer_method=None,
    optimizer_options=None,
):
    df = make_estimation_settings_table(
        config=config,
        objective_value=objective_value,
        weighting_matrix_name=weighting_matrix_name,
        sample_label=sample_label,
        n_obs=n_obs,
        weighting_details=weighting_details,
        optimizer_method=optimizer_method,
        optimizer_options=optimizer_options,
    )
    return _save_table(df, filename_stem, decimals=decimals)


def make_identification_detailed_table(moment_labels, param_names, jacobian):
    jacobian = np.asarray(jacobian, dtype=float)

    if jacobian.shape != (len(moment_labels), len(param_names)):
        raise ValueError(
            f"Jacobian has shape {jacobian.shape}, expected "
            f"({len(moment_labels)}, {len(param_names)})."
        )

    pretty_moments = [_pretty_moment_label(x) for x in moment_labels]
    pretty_params = [_pretty_param_label(x) for x in param_names]

    rows = []
    for i, moment in enumerate(pretty_moments):
        row = {"moment": moment}
        for j, param in enumerate(pretty_params):
            row[f"d(moment)/d({param})"] = float(jacobian[i, j])
        rows.append(row)

    return pd.DataFrame(rows)


def save_identification_detailed_table(
    moment_labels,
    param_names,
    jacobian,
    decimals: int = 6,
    filename_stem: str = "identification_jacobian",
):
    df = make_identification_detailed_table(
        moment_labels=moment_labels,
        param_names=param_names,
        jacobian=jacobian,
    )
    return _save_table(df, filename_stem, decimals=decimals)


def make_identification_summary_table(
    moment_labels,
    param_names,
    jacobian,
    bread_condition_number=None,
):
    jacobian = np.asarray(jacobian, dtype=float)

    if jacobian.shape != (len(moment_labels), len(param_names)):
        raise ValueError(
            f"Jacobian has shape {jacobian.shape}, expected "
            f"({len(moment_labels)}, {len(param_names)})."
        )

    rows = []
    for j, param in enumerate(param_names):
        column = jacobian[:, j]
        abs_column = np.abs(column)
        best_idx = int(np.argmax(abs_column))

        rows.append(
            {
                "parameter": _pretty_param_label(param),
                "jacobian_column_norm": float(np.linalg.norm(column)),
                "largest_loading_moment": _pretty_moment_label(moment_labels[best_idx]),
                "largest_absolute_loading": float(abs_column[best_idx]),
                "bread_condition_number": float(bread_condition_number)
                if bread_condition_number is not None
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def save_identification_summary_table(
    moment_labels,
    param_names,
    jacobian,
    bread_condition_number=None,
    decimals: int = 6,
    filename_stem: str = "identification_summary",
):
    df = make_identification_summary_table(
        moment_labels=moment_labels,
        param_names=param_names,
        jacobian=jacobian,
        bread_condition_number=bread_condition_number,
    )
    return _save_table(df, filename_stem, decimals=decimals)


def make_overidentification_note_table(sample_label, n_moments, n_params):
    dof = int(n_moments) - int(n_params)

    rows = [
        {
            "sample_label": sample_label,
            "n_moments": int(n_moments),
            "n_params": int(n_params),
            "overidentified": bool(dof > 0),
            "degrees_of_freedom": int(dof),
            "formal_j_test_computed": False,
            "note": (
                "Model is overidentified because the number of moments exceeds "
                "the number of estimated parameters. A formal J-test is not "
                "computed here; this table documents the overidentification "
                "status for the report."
            ),
        }
    ]

    return pd.DataFrame(rows)


def save_overidentification_note_table(
    sample_label,
    n_moments,
    n_params,
    filename_stem: str = "overidentification_note",
    decimals: int = 4,
):
    df = make_overidentification_note_table(
        sample_label=sample_label,
        n_moments=n_moments,
        n_params=n_params,
    )
    return _save_table(df, filename_stem, decimals=decimals)


def make_subsample_comparison_table(full_sample, small_firms, large_firms, split_info=None):
    rows = []

    param_names = ["psi", "lambda_external_finance", "rho", "sigma"]

    full_theta = np.asarray(full_sample["theta_hat"], dtype=float)
    small_theta = np.asarray(small_firms["theta_hat"], dtype=float)
    large_theta = np.asarray(large_firms["theta_hat"], dtype=float)

    full_se = np.asarray(full_sample["std_errors"], dtype=float)
    small_se = np.asarray(small_firms["std_errors"], dtype=float)
    large_se = np.asarray(large_firms["std_errors"], dtype=float)

    for i, param in enumerate(param_names):
        rows.append(
            {
                "parameter": _pretty_param_label(param),
                "full_sample_estimate": float(full_theta[i]),
                "full_sample_std_error": float(full_se[i]),
                "small_firms_estimate": float(small_theta[i]),
                "small_firms_std_error": float(small_se[i]),
                "large_firms_estimate": float(large_theta[i]),
                "large_firms_std_error": float(large_se[i]),
                "large_minus_small": float(large_theta[i] - small_theta[i]),
            }
        )

    df = pd.DataFrame(rows)

    if split_info is not None:
        df.attrs["split_info"] = split_info

    return df


def save_subsample_comparison_table(
    full_sample,
    small_firms,
    large_firms,
    split_info=None,
    decimals: int = 4,
):
    df = make_subsample_comparison_table(
        full_sample=full_sample,
        small_firms=small_firms,
        large_firms=large_firms,
        split_info=split_info,
    )
    return _save_table(df, "subsample_parameter_comparison", decimals=decimals)
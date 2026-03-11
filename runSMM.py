from pathlib import Path
import copy

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import build_compustat, load_clean_data
from src.dp_solver import solve_investment_dp
from src.model import get_fixed_params
from src.moments import compute_moments, moment_names
from src.reporting import (
    save_summary_statistics_table,
    save_moment_comparison_table,
    save_parameter_table,
    save_subsample_comparison_table,
)
from src.simulate import simulate_panel
from src.diagnostics_plots import (
    plot_investment_policy_heatmap,
    plot_capital_distribution,
)
from src.smm import (
    estimate_smm,
    make_weighting_matrix,
)
from src.jacobian import smm_standard_errors


TABLE_DIR = Path("results/tables")


def build_policy_summary_table(solution, theta, label):
    """
    Summarize solved policy functions at representative grid points.
    """
    theta = np.asarray(theta, dtype=float)

    k_grid = solution["k_grid"]
    p_grid = solution["p_grid"]
    z_grid = solution["z_grid"]

    policy_i = solution["policy_investment"]
    policy_kprime = solution["policy_kprime"]
    policy_pprime = solution["policy_pprime"]

    k_indices = [0, len(k_grid) // 2, len(k_grid) - 1]
    p_indices = [0, len(p_grid) // 2, len(p_grid) - 1]
    z_indices = [0, len(z_grid) // 2, len(z_grid) - 1]

    rows = []

    for ik in k_indices:
        for ip in p_indices:
            for iz in z_indices:
                rows.append(
                    {
                        "scenario": label,
                        "rho": float(theta[2]),
                        "sigma": float(theta[3]),
                        "psi": float(theta[0]),
                        "lambda_external_finance": float(theta[1]),
                        "k_index": int(ik),
                        "p_index": int(ip),
                        "z_index": int(iz),
                        "k": float(k_grid[ik]),
                        "cash_state": float(p_grid[ip]),
                        "z": float(z_grid[iz]),
                        "investment_policy": float(policy_i[ik, ip, iz]),
                        "kprime_policy": float(policy_kprime[ik, ip, iz]),
                        "cash_next_policy": float(policy_pprime[ik, ip, iz]),
                    }
                )

    return pd.DataFrame(rows)


def print_vector_with_labels(title, labels, values):
    """
    Pretty-print a moment vector alongside its labels.
    """
    print(f"\n{title}")
    for name, value in zip(labels, np.asarray(values, dtype=float)):
        print(f"  {name:<28} {value: .6f}")


def print_solver_diagnostics(solution):
    """
    Print core DP diagnostics.
    """
    print("\nSolver diagnostics")
    print(f"  converged                {solution['converged']}")
    print(f"  iterations               {solution['n_iter']}")
    print(f"  bellman_sup_norm         {solution['max_diff']:.8f}")
    print(f"  share_lower_bound        {solution['share_lower_bound']:.6f}")
    print(f"  share_upper_bound        {solution['share_upper_bound']:.6f}")
    print(f"  share_any_bound          {solution['share_any_bound']:.6f}")
    print(f"  share_interior           {solution['share_interior']:.6f}")


def estimation_spec(config):
    """
    Starting point for the derivative-free capped SMM test.
    """
    theta0 = np.asarray(config["theta0"], dtype=float)

    return {
        "label": "four_parameter_smm",
        "theta0": theta0,
    }


def save_estimation_results(
    moment_labels,
    theta_hat,
    m_data,
    m_sim,
    std_errors=None,
    prefix="",
):
    """
    Save compact estimation output tables.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    prefix_str = f"{prefix}_" if prefix else ""

    param_names = [
        "psi",
        "lambda_external_finance",
        "rho",
        "sigma",
    ]

    param_table = pd.DataFrame(
        {
            "parameter": param_names,
            "estimate": np.asarray(theta_hat, dtype=float),
        }
    )

    if std_errors is not None:
        param_table["std_error"] = np.asarray(std_errors, dtype=float)

    param_csv = TABLE_DIR / f"{prefix_str}smm_parameter_estimates.csv"
    param_table.to_csv(param_csv, index=False)

    moment_table = pd.DataFrame(
        {
            "moment": moment_labels,
            "data": np.asarray(m_data, dtype=float),
            "simulated": np.asarray(m_sim, dtype=float),
            "gap": np.asarray(m_sim, dtype=float) - np.asarray(m_data, dtype=float),
            "abs_gap": np.abs(
                np.asarray(m_sim, dtype=float) - np.asarray(m_data, dtype=float)
            ),
        }
    )
    moment_csv = TABLE_DIR / f"{prefix_str}smm_moment_fit.csv"
    moment_table.to_csv(moment_csv, index=False)

    return param_csv, moment_csv, param_table, moment_table


def save_policy_table(df, filename):
    """
    Save a policy summary table to CSV.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def build_size_tercile_subsamples(df):
    """
    Build bottom-tercile and top-tercile size subsamples using total assets.

    The cleaned Compustat file uses the standard Compustat name:
        at = total assets
    """
    if "at" not in df.columns:
        raise KeyError("The cleaned dataset must contain an 'at' (total assets) column.")

    size_series = pd.to_numeric(df["at"], errors="coerce")
    valid = df.loc[size_series.notna()].copy()
    valid["_size_at_for_split"] = pd.to_numeric(valid["at"], errors="coerce")

    q33 = float(valid["_size_at_for_split"].quantile(1.0 / 3.0))
    q67 = float(valid["_size_at_for_split"].quantile(2.0 / 3.0))

    small_df = valid.loc[valid["_size_at_for_split"] <= q33].copy()
    large_df = valid.loc[valid["_size_at_for_split"] >= q67].copy()

    small_df = small_df.drop(columns=["_size_at_for_split"])
    large_df = large_df.drop(columns=["_size_at_for_split"])

    info = {
        "size_variable": "at",
        "q33": q33,
        "q67": q67,
        "n_small": int(len(small_df)),
        "n_large": int(len(large_df)),
    }

    return small_df, large_df, info


def run_single_estimation(
    df,
    config,
    sample_label,
    theta0,
    save_outputs=True,
    make_plots=False,
):
    """
    Run one full estimation block on one dataset.
    """
    moment_labels = moment_names(config)
    m_data = compute_moments(df, config)

    config_run = copy.deepcopy(config)
    config_run["theta0"] = np.asarray(theta0, dtype=float).tolist()

    print("\n========================================")
    print(f"RUNNING SAMPLE: {sample_label}")
    print("========================================")
    print(f"Sample observations: {len(df)}")

    print_vector_with_labels(
        title=f"{sample_label} empirical target moments",
        labels=moment_labels,
        values=m_data,
    )

    fixed_params_run = get_fixed_params(config_run, theta=theta0)

    print("\n----------------------------------------")
    print(f"{sample_label} baseline DP check")
    print("----------------------------------------")

    baseline_solution = solve_investment_dp(
        theta=np.asarray(theta0, dtype=float),
        config=config_run,
        fixed_params=fixed_params_run,
    )

    baseline_sim_df = simulate_panel(theta0, config_run, solution=baseline_solution)
    baseline_m_sim = compute_moments(baseline_sim_df, config_run)

    if make_plots:
        plot_investment_policy_heatmap(baseline_solution)
        plot_capital_distribution(baseline_sim_df)

    print_solver_diagnostics(baseline_solution)

    W, weighting_details = make_weighting_matrix(
        df=df,
        config=config_run,
        return_details=True,
    )

    print("\nWeighting matrix constructed.")
    print(f"  ridge_scale              {weighting_details['ridge_scale']:.8e}")
    print(f"  condition_number         {weighting_details['condition_number']:.6e}")
    print(f"  used_pinv                {weighting_details['used_pinv']}")

    bounds = config_run["bounds"]

    print("\n----------------------------------------")
    print(f"{sample_label} SMM estimation")
    print("----------------------------------------")

    result = estimate_smm(
        theta0=np.asarray(theta0, dtype=float),
        bounds=bounds,
        m_data=m_data,
        W=W,
        config=config_run,
    )

    theta_hat = np.asarray(result["theta_hat"], dtype=float)
    m_sim_hat = np.asarray(result["m_sim"], dtype=float)
    solution_hat = result["solution_hat"]

    print("\nOptimization results")
    print(f"  converged                {result['success']}")
    print(f"  message                  {result['message']}")
    print(f"  iterations               {result['n_iter']}")
    print(f"  function evaluations     {result['n_fun_eval']}")
    print(f"  initial objective        {result['objective_initial']:.8f}")
    print(f"  final objective          {result['objective_value']:.8f}")
    print(f"  improvement              {result['objective_improvement']:.8f}")

    print("\nEstimated parameters")
    for name, val in zip(
        ["psi", "lambda_external_finance", "rho", "sigma"],
        theta_hat,
    ):
        print(f"  {name:<28} {val:.6f}")

    print_vector_with_labels(
        title=f"{sample_label} simulated moments at estimated parameters",
        labels=moment_labels,
        values=m_sim_hat,
    )

    print_vector_with_labels(
        title=f"{sample_label} moment gaps at estimated parameters (simulated - data)",
        labels=moment_labels,
        values=(m_sim_hat - m_data),
    )

    print("\nEstimated-parameter DP diagnostics")
    print_solver_diagnostics(solution_hat)

    print("\nComputing standard errors")
    se_results = smm_standard_errors(
        theta_hat=theta_hat,
        W=W,
        df=df,
        config=config_run,
    )
    std_errors = np.asarray(se_results["std_errors"], dtype=float)

    print("Standard error diagnostics")
    print(
        f"  simulation_to_data_ratio {se_results['simulation_to_data_ratio']:.6f}"
    )
    print(
        f"  simulation_adjustment    {se_results['simulation_adjustment']:.6f}"
    )
    print(
        f"  bread_condition_number   {se_results['bread_condition_number']:.6e}"
    )

    prefix = sample_label.lower().replace(" ", "_")

    saved_paths = {}

    if save_outputs:
        param_csv, moment_csv, param_table, moment_table = save_estimation_results(
            moment_labels=moment_labels,
            theta_hat=theta_hat,
            m_data=m_data,
            m_sim=m_sim_hat,
            std_errors=std_errors,
            prefix=prefix,
        )

        pretty_param_df, pretty_param_csv, pretty_param_tex = save_parameter_table(
            theta_hat=theta_hat,
            param_names=["psi", "lambda_external_finance", "rho", "sigma"],
            std_errors=std_errors,
            decimals=4,
            filename_stem=f"{prefix}_parameter_estimates",
        )

        estimated_policy_summary_df = build_policy_summary_table(
            solution=solution_hat,
            theta=theta_hat,
            label=f"{prefix}_estimated_theta_hat",
        )
        estimated_policy_csv = save_policy_table(
            estimated_policy_summary_df,
            filename=f"{prefix}_estimated_policy_summary.csv",
        )

        saved_paths = {
            "param_csv": param_csv,
            "moment_csv": moment_csv,
            "pretty_param_csv": pretty_param_csv,
            "pretty_param_tex": pretty_param_tex,
            "estimated_policy_csv": estimated_policy_csv,
        }
    else:
        param_table = pd.DataFrame()
        moment_table = pd.DataFrame()
        pretty_param_df = pd.DataFrame()
        estimated_policy_summary_df = pd.DataFrame()

    return {
        "sample_label": sample_label,
        "n_obs": int(len(df)),
        "moment_labels": moment_labels,
        "m_data": np.asarray(m_data, dtype=float),
        "theta_hat": theta_hat,
        "m_sim_hat": m_sim_hat,
        "std_errors": std_errors,
        "result": result,
        "solution_hat": solution_hat,
        "se_results": se_results,
        "param_table": pretty_param_df if save_outputs else param_table,
        "moment_table": moment_table,
        "estimated_policy_summary_df": estimated_policy_summary_df,
        "saved_paths": saved_paths,
    }


def main():
    config = load_config("settings.json")
    debug_mode = bool(config.get("debug_mode", False))
    validation_mode = config.get("validation_mode", "fast")

    print("\n========================================")
    print("CONFIG LOAD CHECK")
    print("========================================")
    print(f"Loaded theta0 from config: {config['theta0']}")
    print(f"Loaded bounds from config: {config['bounds']}")
    print(
        "Loaded optimizer method from config: "
        f"{config.get('smm_optimizer_method', 'Nelder-Mead')}"
    )
    print(
        "Loaded optimizer options from config: "
        f"{config.get('smm_optimizer_options', {})}"
    )

    if config.get("rebuild_data", False):
        build_compustat(
            raw_path=config["raw_data_path"],
            clean_path=config["clean_data_path"],
            config=config,
        )

    df = load_clean_data(config["clean_data_path"])

    summary_df, summary_csv, summary_tex = save_summary_statistics_table(
        df=df,
        decimals=4,
    )

    spec = estimation_spec(config)
    theta0 = np.asarray(spec["theta0"], dtype=float)

    print("\n========================================")
    print("DERIVATIVE-FREE CAPPED SMM TEST RUN")
    print("========================================")
    print(f"Debug mode: {debug_mode}")
    print(f"Validation mode: {validation_mode}")
    print(f"Number of targeted moments: {config['n_moments']}")
    print(f"Targeted moments: {moment_names(config)}")
    print(f"Starting theta0: {theta0}")
    print(f"Bounds: {config['bounds']}")
    print(
        "Optimizer method: "
        f"{config.get('smm_optimizer_method', 'Nelder-Mead')}"
    )
    print(
        "Optimizer options: "
        f"{config.get('smm_optimizer_options', {})}"
    )

    full_results = run_single_estimation(
        df=df,
        config=config,
        sample_label="full_sample",
        theta0=theta0,
        save_outputs=True,
        make_plots=True,
    )

    small_df, large_df, split_info = build_size_tercile_subsamples(df)

    print("\n========================================")
    print("SIZE TERCILE SUBSAMPLES")
    print("========================================")
    print(f"Size variable: {split_info['size_variable']}")
    print(f"33rd percentile: {split_info['q33']:.6f}")
    print(f"67th percentile: {split_info['q67']:.6f}")
    print(f"Small-firm observations: {split_info['n_small']}")
    print(f"Large-firm observations: {split_info['n_large']}")

    small_results = run_single_estimation(
        df=small_df,
        config=config,
        sample_label="small_firms",
        theta0=theta0,
        save_outputs=True,
        make_plots=False,
    )

    large_results = run_single_estimation(
        df=large_df,
        config=config,
        sample_label="large_firms",
        theta0=theta0,
        save_outputs=True,
        make_plots=False,
    )

    subsample_comparison_df, subsample_comparison_csv, subsample_comparison_tex = (
        save_subsample_comparison_table(
            full_sample=full_results,
            small_firms=small_results,
            large_firms=large_results,
            split_info=split_info,
            decimals=4,
        )
    )

    print("\n========================================")
    print("SUBSAMPLE PARAMETER COMPARISON TABLE")
    print("========================================")
    print(subsample_comparison_df.to_string(index=False))

    print("\n========================================")
    print("SUMMARY STATISTICS TABLE")
    print("========================================")
    print(summary_df.to_string(index=False))

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")
    print(f"Summary statistics CSV: {summary_csv}")
    print(f"Summary statistics TEX: {summary_tex}")

    for label, results_dict in [
        ("Full sample", full_results),
        ("Small firms", small_results),
        ("Large firms", large_results),
    ]:
        print(f"\n{label}")
        for key, value in results_dict["saved_paths"].items():
            print(f"  {key}: {value}")

    print(f"\nSubsample comparison CSV: {subsample_comparison_csv}")
    print(f"Subsample comparison TEX: {subsample_comparison_tex}")

    print("\nRun complete.")


if __name__ == "__main__":
    main()
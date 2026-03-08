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


TABLE_DIR = Path("results/tables")


def build_policy_summary_table(solution, theta, label, rho_value, sigma_value):
    """
    Summarize solved policy functions at representative grid points.

    Current structural stage:
    - state = (k, p, z)
    - controls = (k', p')
    - theta[0] = psi
    - theta[1] = lambda
    """
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
                        "rho": float(rho_value),
                        "sigma": float(sigma_value),
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


def estimation_spec():
    """
    Starting point for the derivative-free capped SMM test.

    This uses the best local point from the objective-surface test.
    """
    return {
        "label": "best_local_start_for_derivative_free_smm",
        "theta0": np.array([0.15, 6.00], dtype=float),
        "rho": 0.90,
        "sigma": 0.40,
    }


def save_estimation_results(moment_labels, theta_hat, m_data, m_sim):
    """
    Save compact estimation output tables.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    param_table = pd.DataFrame(
        {
            "parameter": ["psi", "lambda_external_finance"],
            "estimate": np.asarray(theta_hat, dtype=float),
        }
    )
    param_csv = TABLE_DIR / "smm_parameter_estimates.csv"
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
    moment_csv = TABLE_DIR / "smm_moment_fit.csv"
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


def main():
    config = load_config("settings.json")
    debug_mode = bool(config.get("debug_mode", False))
    validation_mode = config.get("validation_mode", "fast")

    if config.get("rebuild_data", False):
        build_compustat(
            raw_path=config["raw_data_path"],
            clean_path=config["clean_data_path"],
            config=config,
        )

    df = load_clean_data(config["clean_data_path"])

    moment_labels = moment_names(config)
    m_data = compute_moments(df, config)

    summary_df, summary_csv, summary_tex = save_summary_statistics_table(
        df=df,
        decimals=4,
    )

    spec = estimation_spec()
    label = spec["label"]
    theta0 = np.asarray(spec["theta0"], dtype=float)
    rho_value = float(spec["rho"])
    sigma_value = float(spec["sigma"])

    print("\n========================================")
    print("DERIVATIVE-FREE CAPPED SMM TEST RUN")
    print("========================================")
    print(f"Debug mode: {debug_mode}")
    print(f"Validation mode: {validation_mode}")
    print(f"Number of targeted moments: {config['n_moments']}")
    print(f"Targeted moments: {moment_labels}")
    print(f"Starting theta0: {theta0}")
    print(f"Fixed rho: {rho_value:.4f}")
    print(f"Fixed sigma: {sigma_value:.4f}")

    print_vector_with_labels(
        title="Empirical target moments",
        labels=moment_labels,
        values=m_data,
    )

    config_run = copy.deepcopy(config)
    config_run["theta0"] = theta0.tolist()
    config_run["model"]["rho"] = rho_value
    config_run["model"]["sigma"] = sigma_value

    print("\n========================================")
    print("BASELINE DP CHECK AT STARTING VALUES")
    print("========================================")

    fixed_params_run = get_fixed_params(config_run)

    baseline_solution = solve_investment_dp(
        theta=theta0,
        config=config_run,
        fixed_params=fixed_params_run,
    )

    baseline_sim_df = simulate_panel(theta0, config_run, solution=baseline_solution)
    baseline_m_sim = compute_moments(baseline_sim_df, config_run)

    plot_investment_policy_heatmap(baseline_solution)
    plot_capital_distribution(baseline_sim_df)

    print_solver_diagnostics(baseline_solution)

    print_vector_with_labels(
        title="Starting-value simulated moments",
        labels=moment_labels,
        values=baseline_m_sim,
    )

    print_vector_with_labels(
        title="Starting-value moment gaps (simulated - data)",
        labels=moment_labels,
        values=np.asarray(baseline_m_sim, dtype=float) - np.asarray(m_data, dtype=float),
    )

    baseline_moment_table_df, baseline_moment_csv, baseline_moment_tex = (
        save_moment_comparison_table(
            moment_labels=moment_labels,
            m_data=m_data,
            m_sim=baseline_m_sim,
            decimals=4,
        )
    )

    baseline_policy_summary_df = build_policy_summary_table(
        solution=baseline_solution,
        theta=theta0,
        label=label,
        rho_value=rho_value,
        sigma_value=sigma_value,
    )
    baseline_policy_csv = save_policy_table(
        baseline_policy_summary_df,
        filename="baseline_policy_summary.csv",
    )

    print("\nStarting-value moment comparison table")
    print(baseline_moment_table_df.to_string(index=False))

    print("\n========================================")
    print("CONSTRUCTING WEIGHTING MATRIX")
    print("========================================")

    W, weighting_details = make_weighting_matrix(
        df=df,
        config=config_run,
        return_details=True,
    )

    print("Weighting matrix constructed.")
    print(f"  ridge_scale              {weighting_details['ridge_scale']:.8e}")
    print(f"  condition_number         {weighting_details['condition_number']:.6e}")
    print(f"  used_pinv                {weighting_details['used_pinv']}")

    bounds = config_run["bounds"]

    print("\n========================================")
    print("STARTING DERIVATIVE-FREE CAPPED SMM")
    print("========================================")
    print(f"Initial parameters: {theta0}")
    print(f"Optimizer method: {config_run.get('smm_optimizer_method', 'Powell')}")
    print(f"Optimizer options: {config_run.get('smm_optimizer_options', {})}")

    result = estimate_smm(
        theta0=theta0,
        bounds=bounds,
        m_data=m_data,
        W=W,
        config=config_run,
    )

    theta_hat = np.asarray(result["theta_hat"], dtype=float)
    m_sim_hat = np.asarray(result["m_sim"], dtype=float)
    solution_hat = result["solution_hat"]

    print("\n========================================")
    print("OPTIMIZATION RESULTS")
    print("========================================")
    print(f"Converged: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Iterations: {result['n_iter']}")
    print(f"Function evaluations: {result['n_fun_eval']}")
    print(f"Initial objective value: {result['objective_initial']:.8f}")
    print(f"Final objective value:   {result['objective_value']:.8f}")
    print(f"Objective improvement:   {result['objective_improvement']:.8f}")

    print("\nEstimated parameters")
    for name, val in zip(["psi", "lambda_external_finance"], theta_hat):
        print(f"  {name:<28} {val:.6f}")

    print_vector_with_labels(
        title="Simulated moments at estimated parameters",
        labels=moment_labels,
        values=m_sim_hat,
    )

    print_vector_with_labels(
        title="Moment gaps at estimated parameters (simulated - data)",
        labels=moment_labels,
        values=(m_sim_hat - m_data),
    )

    print("\n========================================")
    print("ESTIMATED-PARAMETER DP DIAGNOSTICS")
    print("========================================")
    print_solver_diagnostics(solution_hat)

    param_csv, moment_csv, param_table, moment_table = save_estimation_results(
        moment_labels=moment_labels,
        theta_hat=theta_hat,
        m_data=m_data,
        m_sim=m_sim_hat,
    )

    estimated_policy_summary_df = build_policy_summary_table(
        solution=solution_hat,
        theta=theta_hat,
        label="estimated_theta_hat",
        rho_value=rho_value,
        sigma_value=sigma_value,
    )
    estimated_policy_csv = save_policy_table(
        estimated_policy_summary_df,
        filename="estimated_policy_summary.csv",
    )

    print("\n========================================")
    print("PARAMETER ESTIMATES TABLE")
    print("========================================")
    print(param_table.to_string(index=False))

    print("\n========================================")
    print("ESTIMATED MOMENT FIT TABLE")
    print("========================================")
    print(moment_table.to_string(index=False))

    print("\n========================================")
    print("BASELINE POLICY SUMMARY TABLE")
    print("========================================")
    print(baseline_policy_summary_df.to_string(index=False))

    print("\n========================================")
    print("ESTIMATED POLICY SUMMARY TABLE")
    print("========================================")
    print(estimated_policy_summary_df.to_string(index=False))

    print("\n========================================")
    print("SUMMARY STATISTICS TABLE")
    print("========================================")
    print(summary_df.to_string(index=False))

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")
    print(f"Summary statistics CSV: {summary_csv}")
    print(f"Summary statistics TEX: {summary_tex}")
    print(f"Starting-value moment comparison CSV: {baseline_moment_csv}")
    print(f"Starting-value moment comparison TEX: {baseline_moment_tex}")
    print(f"Baseline policy summary CSV: {baseline_policy_csv}")
    print(f"Estimated policy summary CSV: {estimated_policy_csv}")
    print(f"Parameter estimates CSV: {param_csv}")
    print(f"Moment fit CSV: {moment_csv}")

    print("\nRun complete.")


if __name__ == "__main__":
    main()
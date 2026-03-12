from pathlib import Path
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src.data import build_compustat, load_clean_data
from src.dp_solver import solve_investment_dp
from src.model import get_fixed_params
from src.moments import compute_moments, moment_names
from src.reporting import save_summary_statistics_table
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
FIGURE_DIR = Path("results/figures")


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
    Starting point for diagnostics.
    """
    theta0 = np.asarray(config["theta0"], dtype=float)

    return {
        "label": "four_parameter_smm",
        "theta0": theta0,
    }


def save_diagnostic_table(df, filename):
    """
    Save a generic diagnostics table to CSV.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def _distance_to_bounds(theta, bounds):
    """
    Compute distance of each parameter to its lower and upper bounds.
    """
    theta = np.asarray(theta, dtype=float)

    rows = []
    for i, (value, bound) in enumerate(zip(theta, bounds)):
        lower = float(bound[0])
        upper = float(bound[1])
        rows.append(
            {
                "param_index": i,
                "theta_value": float(value),
                "lower_bound": lower,
                "upper_bound": upper,
                "distance_to_lower": float(value - lower),
                "distance_to_upper": float(upper - value),
            }
        )
    return pd.DataFrame(rows)


def _build_optimizer_options(base_config, maxiter=None, maxfev=None):
    """
    Build optimizer options from the base config, overriding maxiter/maxfev.
    """
    options = copy.deepcopy(base_config.get("smm_optimizer_options", {}))

    if maxiter is not None:
        options["maxiter"] = int(maxiter)
    if maxfev is not None:
        options["maxfev"] = int(maxfev)

    return options


def _blank_se_results(n_params, n_moments):
    """
    Placeholder standard-error results for fast diagnostics.
    """
    return {
        "jacobian": np.full((n_moments, n_params), np.nan, dtype=float),
        "vcov": np.full((n_params, n_params), np.nan, dtype=float),
        "std_errors": np.full(n_params, np.nan, dtype=float),
        "t_stats": np.full(n_params, np.nan, dtype=float),
        "simulation_to_data_ratio": np.nan,
        "simulation_adjustment": np.nan,
        "bread": np.full((n_params, n_params), np.nan, dtype=float),
        "bread_condition_number": np.nan,
    }


def _apply_dp_overrides(config, dp_overrides=None):
    """
    Return a copy of config with selected dp settings overridden.
    """
    config_run = copy.deepcopy(config)
    if dp_overrides:
        config_run.setdefault("dp", {})
        for key, value in dp_overrides.items():
            config_run["dp"][key] = value
    return config_run


def _apply_model_overrides(config, model_overrides=None):
    """
    Return a copy of config with selected model settings overridden.
    """
    config_run = copy.deepcopy(config)
    if model_overrides:
        config_run.setdefault("model", {})
        for key, value in model_overrides.items():
            config_run["model"][key] = value
    return config_run


def _apply_simulation_overrides(config, simulation_overrides=None):
    """
    Return a copy of config with selected simulation settings overridden.
    """
    config_run = copy.deepcopy(config)
    if simulation_overrides:
        config_run.setdefault("simulation", {})
        for key, value in simulation_overrides.items():
            config_run["simulation"][key] = value
    return config_run


def _apply_theta_overrides(theta, theta_overrides=None):
    """
    Return a copy of theta with named overrides applied.
    """
    theta_run = np.asarray(theta, dtype=float).copy()

    if not theta_overrides:
        return theta_run

    index_map = {
        "psi": 0,
        "lambda_external_finance": 1,
        "lam": 1,
        "rho": 2,
        "sigma": 3,
    }

    for key, value in theta_overrides.items():
        if key not in index_map:
            raise KeyError(
                f"Unknown theta override '{key}'. "
                f"Allowed keys: {list(index_map.keys())}"
            )
        theta_run[index_map[key]] = float(value)

    return theta_run


def save_layer4_policy_plot(policy_rows, filename="layer4_policy_comparison.png"):
    """
    Save a comparison plot of investment policy against cash state across
    Layer 4 scenarios at representative (k, z).
    """
    if not policy_rows:
        return None

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df_plot = pd.DataFrame(policy_rows)

    plt.figure(figsize=(8, 6))

    for scenario_label, subdf in df_plot.groupby("scenario_label"):
        subdf = subdf.sort_values("cash_state")
        plt.plot(
            subdf["cash_state"].to_numpy(dtype=float),
            subdf["investment_policy"].to_numpy(dtype=float),
            label=str(scenario_label),
        )

    rep_k = float(df_plot["representative_k"].iloc[0])
    rep_z = float(df_plot["representative_z"].iloc[0])

    plt.xlabel("Cash state")
    plt.ylabel("Investment policy")
    plt.title(
        "Investment Policy vs Cash Across Layer 4 Scenarios\n"
        f"(representative k = {rep_k:.2f}, z = {rep_z:.2f})"
    )
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved Layer 4 policy comparison plot: {output_path}")
    return output_path


def run_single_estimation(
    df,
    config,
    sample_label,
    theta0,
    make_plots=False,
    run_baseline_check=True,
    compute_standard_errors=True,
):
    """
    Run one estimation block for diagnostics.
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

    if run_baseline_check:
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
        _ = compute_moments(baseline_sim_df, config_run)

        if make_plots:
            plot_investment_policy_heatmap(baseline_solution)
            plot_capital_distribution(baseline_sim_df)

        print_solver_diagnostics(baseline_solution)
    else:
        print("\nBaseline DP check skipped for fast diagnostics.")

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

    if compute_standard_errors:
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
    else:
        print("\nStandard errors skipped for fast diagnostics.")
        se_results = _blank_se_results(
            n_params=len(theta_hat),
            n_moments=len(moment_labels),
        )
        std_errors = np.asarray(se_results["std_errors"], dtype=float)

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
        "weighting_details": weighting_details,
    }


def run_optimizer_budget_diagnostics(df, config, theta0):
    """
    Run optimizer-budget sensitivity diagnostics on the full sample.
    """
    diag_cfg = config.get("optimizer_diagnostics", {})
    budgets = diag_cfg.get("budget_sweep", [])

    if not budgets:
        print("\nNo optimizer budget sweep configurations found.")
        return pd.DataFrame(), None

    rows = []
    moment_labels = moment_names(config)

    for budget in budgets:
        label = str(budget.get("label", "budget"))
        maxiter = int(
            budget.get(
                "maxiter",
                config.get("smm_optimizer_options", {}).get("maxiter", 20),
            )
        )
        maxfev = int(
            budget.get(
                "maxfev",
                config.get("smm_optimizer_options", {}).get("maxfev", 40),
            )
        )

        config_run = copy.deepcopy(config)
        config_run["smm_optimizer_options"] = _build_optimizer_options(
            base_config=config,
            maxiter=maxiter,
            maxfev=maxfev,
        )

        print("\n========================================")
        print(f"OPTIMIZER BUDGET DIAGNOSTIC: {label}")
        print("========================================")
        print(f"maxiter = {maxiter}")
        print(f"maxfev  = {maxfev}")

        results = run_single_estimation(
            df=df,
            config=config_run,
            sample_label=f"diagnostic_{label}",
            theta0=np.asarray(theta0, dtype=float),
            make_plots=bool(diag_cfg.get("make_plots", False)),
            run_baseline_check=bool(diag_cfg.get("run_baseline_check", False)),
            compute_standard_errors=bool(diag_cfg.get("compute_standard_errors", False)),
        )

        theta_hat = np.asarray(results["theta_hat"], dtype=float)
        bounds = config_run["bounds"]
        bound_dist = _distance_to_bounds(theta_hat, bounds)

        row = {
            "diagnostic_label": label,
            "theta0_psi": float(theta0[0]),
            "theta0_lambda": float(theta0[1]),
            "theta0_rho": float(theta0[2]),
            "theta0_sigma": float(theta0[3]),
            "maxiter": maxiter,
            "maxfev": maxfev,
            "success": bool(results["result"]["success"]),
            "message": str(results["result"]["message"]),
            "objective_initial": float(results["result"]["objective_initial"]),
            "objective_value": float(results["result"]["objective_value"]),
            "objective_improvement": float(results["result"]["objective_improvement"]),
            "n_iter": int(results["result"]["n_iter"]),
            "n_fun_eval": int(results["result"]["n_fun_eval"]),
            "psi_hat": float(theta_hat[0]),
            "lambda_hat": float(theta_hat[1]),
            "rho_hat": float(theta_hat[2]),
            "sigma_hat": float(theta_hat[3]),
            "psi_se": float(results["std_errors"][0]) if np.isfinite(results["std_errors"][0]) else np.nan,
            "lambda_se": float(results["std_errors"][1]) if np.isfinite(results["std_errors"][1]) else np.nan,
            "rho_se": float(results["std_errors"][2]) if np.isfinite(results["std_errors"][2]) else np.nan,
            "sigma_se": float(results["std_errors"][3]) if np.isfinite(results["std_errors"][3]) else np.nan,
            "bread_condition_number": float(results["se_results"]["bread_condition_number"])
            if np.isfinite(results["se_results"]["bread_condition_number"])
            else np.nan,
            "dp_converged": bool(results["solution_hat"]["converged"]),
            "dp_iterations": int(results["solution_hat"]["n_iter"]),
            "dp_bellman_sup_norm": float(results["solution_hat"]["max_diff"]),
            "share_lower_bound": float(results["solution_hat"]["share_lower_bound"]),
            "share_upper_bound": float(results["solution_hat"]["share_upper_bound"]),
            "share_any_bound": float(results["solution_hat"]["share_any_bound"]),
            "share_interior": float(results["solution_hat"]["share_interior"]),
            "psi_distance_to_lower": float(bound_dist.loc[0, "distance_to_lower"]),
            "psi_distance_to_upper": float(bound_dist.loc[0, "distance_to_upper"]),
            "lambda_distance_to_lower": float(bound_dist.loc[1, "distance_to_lower"]),
            "lambda_distance_to_upper": float(bound_dist.loc[1, "distance_to_upper"]),
            "rho_distance_to_lower": float(bound_dist.loc[2, "distance_to_lower"]),
            "rho_distance_to_upper": float(bound_dist.loc[2, "distance_to_upper"]),
            "sigma_distance_to_lower": float(bound_dist.loc[3, "distance_to_lower"]),
            "sigma_distance_to_upper": float(bound_dist.loc[3, "distance_to_upper"]),
            "mean_abs_gap": float(np.mean(np.abs(results["m_sim_hat"] - results["m_data"]))),
            "max_abs_gap": float(np.max(np.abs(results["m_sim_hat"] - results["m_data"]))),
        }

        for name, sim_val, data_val in zip(moment_labels, results["m_sim_hat"], results["m_data"]):
            row[f"sim_{name}"] = float(sim_val)
            row[f"data_{name}"] = float(data_val)
            row[f"gap_{name}"] = float(sim_val - data_val)

        rows.append(row)

    df_diag = pd.DataFrame(rows)
    output_path = save_diagnostic_table(df_diag, "optimizer_budget_diagnostics.csv")

    print("\nSaved optimizer budget diagnostics:")
    print(output_path)

    return df_diag, output_path


def run_multistart_diagnostics(df, config):
    """
    Run multi-start optimization diagnostics on the full sample.
    """
    diag_cfg = config.get("optimizer_diagnostics", {})
    starts = diag_cfg.get("multi_start_thetas", [])

    if not starts:
        print("\nNo multi-start configurations found.")
        return pd.DataFrame(), None

    rows = []
    moment_labels = moment_names(config)

    for start in starts:
        label = str(start.get("label", "start"))
        theta0 = np.asarray(start.get("theta0", config["theta0"]), dtype=float)

        print("\n========================================")
        print(f"MULTI-START DIAGNOSTIC: {label}")
        print("========================================")
        print(f"theta0 = {theta0}")

        results = run_single_estimation(
            df=df,
            config=config,
            sample_label=f"multistart_{label}",
            theta0=theta0,
            make_plots=bool(diag_cfg.get("make_plots", False)),
            run_baseline_check=bool(diag_cfg.get("run_baseline_check", False)),
            compute_standard_errors=bool(diag_cfg.get("compute_standard_errors", False)),
        )

        theta_hat = np.asarray(results["theta_hat"], dtype=float)
        bounds = config["bounds"]
        bound_dist = _distance_to_bounds(theta_hat, bounds)

        row = {
            "diagnostic_label": label,
            "theta0_psi": float(theta0[0]),
            "theta0_lambda": float(theta0[1]),
            "theta0_rho": float(theta0[2]),
            "theta0_sigma": float(theta0[3]),
            "success": bool(results["result"]["success"]),
            "message": str(results["result"]["message"]),
            "objective_initial": float(results["result"]["objective_initial"]),
            "objective_value": float(results["result"]["objective_value"]),
            "objective_improvement": float(results["result"]["objective_improvement"]),
            "n_iter": int(results["result"]["n_iter"]),
            "n_fun_eval": int(results["result"]["n_fun_eval"]),
            "psi_hat": float(theta_hat[0]),
            "lambda_hat": float(theta_hat[1]),
            "rho_hat": float(theta_hat[2]),
            "sigma_hat": float(theta_hat[3]),
            "psi_se": float(results["std_errors"][0]) if np.isfinite(results["std_errors"][0]) else np.nan,
            "lambda_se": float(results["std_errors"][1]) if np.isfinite(results["std_errors"][1]) else np.nan,
            "rho_se": float(results["std_errors"][2]) if np.isfinite(results["std_errors"][2]) else np.nan,
            "sigma_se": float(results["std_errors"][3]) if np.isfinite(results["std_errors"][3]) else np.nan,
            "bread_condition_number": float(results["se_results"]["bread_condition_number"])
            if np.isfinite(results["se_results"]["bread_condition_number"])
            else np.nan,
            "dp_converged": bool(results["solution_hat"]["converged"]),
            "dp_iterations": int(results["solution_hat"]["n_iter"]),
            "dp_bellman_sup_norm": float(results["solution_hat"]["max_diff"]),
            "share_lower_bound": float(results["solution_hat"]["share_lower_bound"]),
            "share_upper_bound": float(results["solution_hat"]["share_upper_bound"]),
            "share_any_bound": float(results["solution_hat"]["share_any_bound"]),
            "share_interior": float(results["solution_hat"]["share_interior"]),
            "psi_distance_to_lower": float(bound_dist.loc[0, "distance_to_lower"]),
            "psi_distance_to_upper": float(bound_dist.loc[0, "distance_to_upper"]),
            "lambda_distance_to_lower": float(bound_dist.loc[1, "distance_to_lower"]),
            "lambda_distance_to_upper": float(bound_dist.loc[1, "distance_to_upper"]),
            "rho_distance_to_lower": float(bound_dist.loc[2, "distance_to_lower"]),
            "rho_distance_to_upper": float(bound_dist.loc[2, "distance_to_upper"]),
            "sigma_distance_to_lower": float(bound_dist.loc[3, "distance_to_lower"]),
            "sigma_distance_to_upper": float(bound_dist.loc[3, "distance_to_upper"]),
            "mean_abs_gap": float(np.mean(np.abs(results["m_sim_hat"] - results["m_data"]))),
            "max_abs_gap": float(np.max(np.abs(results["m_sim_hat"] - results["m_data"]))),
        }

        for name, sim_val, data_val in zip(moment_labels, results["m_sim_hat"], results["m_data"]):
            row[f"sim_{name}"] = float(sim_val)
            row[f"data_{name}"] = float(data_val)
            row[f"gap_{name}"] = float(sim_val - data_val)

        rows.append(row)

    df_diag = pd.DataFrame(rows)
    output_path = save_diagnostic_table(df_diag, "optimizer_multistart_diagnostics.csv")

    print("\nSaved multi-start diagnostics:")
    print(output_path)

    return df_diag, output_path


def run_layer3_fixed_theta_diagnostics(df, config):
    """
    Fast Layer 3 diagnostics:
    hold theta fixed and vary grid / bound settings without re-optimizing.
    """
    layer3_cfg = config.get("layer3_diagnostics", {})
    scenarios = layer3_cfg.get("scenarios", [])
    fixed_theta = np.asarray(layer3_cfg.get("fixed_theta", config["theta0"]), dtype=float)
    make_plots = bool(layer3_cfg.get("make_plots", False))

    if not scenarios:
        print("\nNo Layer 3 scenarios found.")
        return pd.DataFrame(), None

    rows = []
    moment_labels = moment_names(config)
    m_data = compute_moments(df, config)

    for scenario in scenarios:
        label = str(scenario.get("label", "scenario"))
        dp_overrides = scenario.get("dp_overrides", {})

        config_run = _apply_dp_overrides(config, dp_overrides=dp_overrides)
        fixed_params_run = get_fixed_params(config_run, theta=fixed_theta)

        print("\n========================================")
        print(f"LAYER 3 FIXED-THETA DIAGNOSTIC: {label}")
        print("========================================")
        print(f"theta = {fixed_theta}")
        print(f"dp_overrides = {dp_overrides}")

        solution = solve_investment_dp(
            theta=fixed_theta,
            config=config_run,
            fixed_params=fixed_params_run,
        )

        sim_df = simulate_panel(fixed_theta, config_run, solution=solution)
        m_sim = compute_moments(sim_df, config_run)

        if make_plots:
            plot_investment_policy_heatmap(solution)
            plot_capital_distribution(sim_df)

        print_solver_diagnostics(solution)
        print_vector_with_labels(
            title=f"{label} simulated moments",
            labels=moment_labels,
            values=m_sim,
        )
        print_vector_with_labels(
            title=f"{label} moment gaps (simulated - data)",
            labels=moment_labels,
            values=(m_sim - m_data),
        )

        row = {
            "scenario_label": label,
            "psi": float(fixed_theta[0]),
            "lambda_external_finance": float(fixed_theta[1]),
            "rho": float(fixed_theta[2]),
            "sigma": float(fixed_theta[3]),
            "k_max": float(config_run["dp"]["k_max"]),
            "p_max": float(config_run["dp"]["p_max"]),
            "investment_max": float(config_run["dp"]["investment_max"]),
            "investment_min": float(config_run["dp"]["investment_min"]),
            "k_grid_size": int(config_run["dp"]["k_grid_size"]),
            "p_grid_size": int(config_run["dp"]["p_grid_size"]),
            "z_grid_size": int(config_run["dp"]["z_grid_size"]),
            "control_grid_size": int(config_run["dp"]["control_grid_size"]),
            "cash_control_grid_size": int(config_run["dp"]["cash_control_grid_size"]),
            "fixed_cost": float(config_run["model"]["fixed_cost"]),
            "production_scale": float(config_run["model"]["production_scale"]),
            "alpha": float(config_run["model"]["alpha"]),
            "delta": float(config_run["model"]["delta"]),
            "cash_return_rate": float(config_run["model"]["cash_return_rate"]),
            "dp_converged": bool(solution["converged"]),
            "dp_iterations": int(solution["n_iter"]),
            "dp_bellman_sup_norm": float(solution["max_diff"]),
            "share_lower_bound": float(solution["share_lower_bound"]),
            "share_upper_bound": float(solution["share_upper_bound"]),
            "share_any_bound": float(solution["share_any_bound"]),
            "share_interior": float(solution["share_interior"]),
            "mean_abs_gap": float(np.mean(np.abs(m_sim - m_data))),
            "max_abs_gap": float(np.max(np.abs(m_sim - m_data))),
        }

        for name, sim_val, data_val in zip(moment_labels, m_sim, m_data):
            row[f"sim_{name}"] = float(sim_val)
            row[f"data_{name}"] = float(data_val)
            row[f"gap_{name}"] = float(sim_val - data_val)

        rows.append(row)

    df_diag = pd.DataFrame(rows)
    output_path = save_diagnostic_table(df_diag, "layer3_fixed_theta_diagnostics.csv")

    print("\nSaved Layer 3 fixed-theta diagnostics:")
    print(output_path)

    return df_diag, output_path


def run_layer4_structural_diagnostics(df, config):
    """
    Fast Layer 4 diagnostics:
    hold the estimation problem fixed, but vary structural ingredients
    without re-optimizing.

    This version supports:
    - theta overrides
    - model overrides
    - simulation overrides
    - dp overrides
    """
    layer4_cfg = config.get("layer4_diagnostics", {})
    scenarios = layer4_cfg.get("scenarios", [])
    base_theta = np.asarray(layer4_cfg.get("fixed_theta", config["theta0"]), dtype=float)
    make_plots = bool(layer4_cfg.get("make_plots", False))
    save_policy_plot = bool(
        layer4_cfg.get("save_policy_plot", layer4_cfg.get("save_lambda_policy_plot", False))
    )

    if not scenarios:
        print("\nNo Layer 4 scenarios found.")
        return pd.DataFrame(), None

    rows = []
    policy_plot_rows = []
    moment_labels = moment_names(config)
    m_data = compute_moments(df, config)

    for scenario in scenarios:
        label = str(scenario.get("label", "scenario"))
        theta_overrides = scenario.get("theta_overrides", {})
        model_overrides = scenario.get("model_overrides", {})
        simulation_overrides = scenario.get("simulation_overrides", {})
        dp_overrides = scenario.get("dp_overrides", {})

        theta_run = _apply_theta_overrides(base_theta, theta_overrides=theta_overrides)
        config_run = _apply_dp_overrides(config, dp_overrides=dp_overrides)
        config_run = _apply_model_overrides(config_run, model_overrides=model_overrides)
        config_run = _apply_simulation_overrides(
            config_run, simulation_overrides=simulation_overrides
        )

        fixed_params_run = get_fixed_params(config_run, theta=theta_run)

        print("\n========================================")
        print(f"LAYER 4 STRUCTURAL DIAGNOSTIC: {label}")
        print("========================================")
        print(f"theta = {theta_run}")
        print(f"theta_overrides = {theta_overrides}")
        print(f"model_overrides = {model_overrides}")
        print(f"simulation_overrides = {simulation_overrides}")
        print(f"dp_overrides = {dp_overrides}")

        solution = solve_investment_dp(
            theta=theta_run,
            config=config_run,
            fixed_params=fixed_params_run,
        )

        sim_df = simulate_panel(theta_run, config_run, solution=solution)
        m_sim = compute_moments(sim_df, config_run)

        if make_plots:
            plot_investment_policy_heatmap(solution)
            plot_capital_distribution(sim_df)

        print_solver_diagnostics(solution)
        print_vector_with_labels(
            title=f"{label} simulated moments",
            labels=moment_labels,
            values=m_sim,
        )
        print_vector_with_labels(
            title=f"{label} moment gaps (simulated - data)",
            labels=moment_labels,
            values=(m_sim - m_data),
        )

        p_grid = solution["p_grid"]
        k_grid = solution["k_grid"]
        z_grid = solution["z_grid"]
        policy_i = solution["policy_investment"]

        ik = len(k_grid) // 2
        iz = len(z_grid) // 2

        for ip, cash_state in enumerate(p_grid):
            policy_plot_rows.append(
                {
                    "scenario_label": label,
                    "lambda_external_finance": float(theta_run[1]),
                    "cash_state": float(cash_state),
                    "investment_policy": float(policy_i[ik, ip, iz]),
                    "representative_k": float(k_grid[ik]),
                    "representative_z": float(z_grid[iz]),
                }
            )

        sim_cfg = config_run.get("simulation", {})

        row = {
            "scenario_label": label,
            "psi": float(theta_run[0]),
            "lambda_external_finance": float(theta_run[1]),
            "rho": float(theta_run[2]),
            "sigma": float(theta_run[3]),
            "fixed_cost": float(config_run["model"]["fixed_cost"]),
            "production_scale": float(config_run["model"]["production_scale"]),
            "alpha": float(config_run["model"]["alpha"]),
            "delta": float(config_run["model"]["delta"]),
            "cash_return_rate": float(config_run["model"]["cash_return_rate"]),
            "initial_cash_ratio": float(sim_cfg.get("initial_cash_ratio", np.nan)),
            "asset_multiplier": float(sim_cfg.get("asset_multiplier", np.nan)),
            "k_max": float(config_run["dp"]["k_max"]),
            "p_max": float(config_run["dp"]["p_max"]),
            "investment_max": float(config_run["dp"]["investment_max"]),
            "investment_min": float(config_run["dp"]["investment_min"]),
            "k_grid_size": int(config_run["dp"]["k_grid_size"]),
            "p_grid_size": int(config_run["dp"]["p_grid_size"]),
            "z_grid_size": int(config_run["dp"]["z_grid_size"]),
            "control_grid_size": int(config_run["dp"]["control_grid_size"]),
            "cash_control_grid_size": int(config_run["dp"]["cash_control_grid_size"]),
            "dp_converged": bool(solution["converged"]),
            "dp_iterations": int(solution["n_iter"]),
            "dp_bellman_sup_norm": float(solution["max_diff"]),
            "share_lower_bound": float(solution["share_lower_bound"]),
            "share_upper_bound": float(solution["share_upper_bound"]),
            "share_any_bound": float(solution["share_any_bound"]),
            "share_interior": float(solution["share_interior"]),
            "mean_abs_gap": float(np.mean(np.abs(m_sim - m_data))),
            "max_abs_gap": float(np.max(np.abs(m_sim - m_data))),
        }

        for name, sim_val, data_val in zip(moment_labels, m_sim, m_data):
            row[f"sim_{name}"] = float(sim_val)
            row[f"data_{name}"] = float(data_val)
            row[f"gap_{name}"] = float(sim_val - data_val)

        rows.append(row)

    df_diag = pd.DataFrame(rows)
    output_path = save_diagnostic_table(df_diag, "layer4_structural_diagnostics.csv")

    policy_plot_csv = None
    policy_plot_png = None

    if policy_plot_rows:
        policy_plot_df = pd.DataFrame(policy_plot_rows)
        policy_plot_csv = save_diagnostic_table(
            policy_plot_df,
            "layer4_policy_plot_data.csv",
        )

        if save_policy_plot:
            policy_plot_png = save_layer4_policy_plot(
                policy_rows=policy_plot_rows,
                filename="layer4_policy_comparison.png",
            )

    print("\nSaved Layer 4 structural diagnostics:")
    print(output_path)
    if policy_plot_csv is not None:
        print(f"Saved Layer 4 policy plot data: {policy_plot_csv}")
    if policy_plot_png is not None:
        print(f"Saved Layer 4 policy plot image: {policy_plot_png}")

    return df_diag, output_path


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

    _, summary_csv, summary_tex = save_summary_statistics_table(
        df=df,
        decimals=4,
    )

    spec = estimation_spec(config)
    theta0 = np.asarray(spec["theta0"], dtype=float)

    print("\n========================================")
    print("DIAGNOSTIC RUN")
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

    any_diagnostics_run = False

    diag_cfg = config.get("optimizer_diagnostics", {})
    diagnostics_enabled = bool(diag_cfg.get("enabled", False))

    if diagnostics_enabled:
        any_diagnostics_run = True
        diagnostics_only = bool(diag_cfg.get("diagnostics_only", False))

        print("\n========================================")
        print("LAYER 2 OPTIMIZATION DIAGNOSTICS")
        print("========================================")
        print(f"Diagnostics only mode: {diagnostics_only}")
        print(f"Run baseline check in diagnostics: {bool(diag_cfg.get('run_baseline_check', False))}")
        print(f"Compute standard errors in diagnostics: {bool(diag_cfg.get('compute_standard_errors', False))}")
        print(f"Make plots in diagnostics: {bool(diag_cfg.get('make_plots', False))}")

        if bool(diag_cfg.get("run_budget_sweep", False)):
            run_optimizer_budget_diagnostics(
                df=df,
                config=config,
                theta0=theta0,
            )

        if bool(diag_cfg.get("run_multi_start", False)):
            run_multistart_diagnostics(
                df=df,
                config=config,
            )

    layer3_cfg = config.get("layer3_diagnostics", {})
    layer3_enabled = bool(layer3_cfg.get("enabled", False))

    if layer3_enabled:
        any_diagnostics_run = True
        layer3_diagnostics_only = bool(layer3_cfg.get("diagnostics_only", False))

        print("\n========================================")
        print("LAYER 3 GRID / BOUND DIAGNOSTICS")
        print("========================================")
        print(f"Diagnostics only mode: {layer3_diagnostics_only}")
        print(f"Fixed theta: {layer3_cfg.get('fixed_theta', config['theta0'])}")

        run_layer3_fixed_theta_diagnostics(
            df=df,
            config=config,
        )

    layer4_cfg = config.get("layer4_diagnostics", {})
    layer4_enabled = bool(layer4_cfg.get("enabled", False))

    if layer4_enabled:
        any_diagnostics_run = True
        layer4_diagnostics_only = bool(layer4_cfg.get("diagnostics_only", False))

        print("\n========================================")
        print("LAYER 4 STRUCTURAL DIAGNOSTICS")
        print("========================================")
        print(f"Diagnostics only mode: {layer4_diagnostics_only}")
        print(f"Fixed theta: {layer4_cfg.get('fixed_theta', config['theta0'])}")

        run_layer4_structural_diagnostics(
            df=df,
            config=config,
        )

    if not any_diagnostics_run:
        print("\nNo diagnostics blocks are enabled in settings.json.")
        print("Nothing to run.")
        return

    print("\nSummary statistics CSV:", summary_csv)
    print("Summary statistics TEX:", summary_tex)
    print("\nDiagnostic run complete.")


if __name__ == "__main__":
    main()
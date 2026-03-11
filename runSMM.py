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
    save_parameter_table,
    save_subsample_comparison_table,
    save_estimation_settings_table,
    save_identification_detailed_table,
    save_identification_summary_table,
    save_overidentification_note_table,
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

    moment_label_map = {
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
    pretty_moment_labels = [moment_label_map.get(x, x) for x in moment_labels]

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
            "moment": pretty_moment_labels,
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


def save_diagnostic_table(df, filename):
    """
    Save a generic diagnostics table to CSV.
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


def run_single_estimation(
    df,
    config,
    sample_label,
    theta0,
    save_outputs=True,
    make_plots=False,
    run_baseline_check=True,
    compute_standard_errors=True,
):
    """
    Run one full estimation block on one dataset.

    Fast-diagnostics mode:
    - run_baseline_check = False
    - compute_standard_errors = False

    This preserves the main estimation pipeline while avoiding expensive
    steps that are unnecessary during optimizer / grid debugging.
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

        estimation_settings_df, estimation_settings_csv, estimation_settings_tex = (
            save_estimation_settings_table(
                config=config_run,
                objective_value=result["objective_value"],
                weighting_matrix_name="Efficient SMM",
                decimals=4,
                filename_stem=f"{prefix}_estimation_settings",
                sample_label=sample_label,
                n_obs=int(len(df)),
                weighting_details=weighting_details,
                optimizer_method=config_run.get("smm_optimizer_method", "Nelder-Mead"),
                optimizer_options=config_run.get("smm_optimizer_options", {}),
            )
        )

        identification_detail_df, identification_detail_csv, identification_detail_tex = (
            save_identification_detailed_table(
                moment_labels=moment_labels,
                param_names=["psi", "lambda_external_finance", "rho", "sigma"],
                jacobian=se_results["jacobian"],
                decimals=6,
                filename_stem=f"{prefix}_identification_jacobian",
            )
        )

        identification_summary_df, identification_summary_csv, identification_summary_tex = (
            save_identification_summary_table(
                moment_labels=moment_labels,
                param_names=["psi", "lambda_external_finance", "rho", "sigma"],
                jacobian=se_results["jacobian"],
                bread_condition_number=se_results["bread_condition_number"],
                decimals=6,
                filename_stem=f"{prefix}_identification_summary",
            )
        )

        overid_note_df, overid_note_csv, overid_note_tex = save_overidentification_note_table(
            sample_label=sample_label,
            n_moments=len(moment_labels),
            n_params=len(theta_hat),
            filename_stem=f"{prefix}_overidentification_note",
            decimals=4,
        )

        saved_paths = {
            "param_csv": param_csv,
            "moment_csv": moment_csv,
            "pretty_param_csv": pretty_param_csv,
            "pretty_param_tex": pretty_param_tex,
            "estimated_policy_csv": estimated_policy_csv,
            "estimation_settings_csv": estimation_settings_csv,
            "estimation_settings_tex": estimation_settings_tex,
            "identification_detail_csv": identification_detail_csv,
            "identification_detail_tex": identification_detail_tex,
            "identification_summary_csv": identification_summary_csv,
            "identification_summary_tex": identification_summary_tex,
            "overid_note_csv": overid_note_csv,
            "overid_note_tex": overid_note_tex,
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
            save_outputs=False,
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
            save_outputs=False,
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

    diag_cfg = config.get("optimizer_diagnostics", {})
    diagnostics_enabled = bool(diag_cfg.get("enabled", False))
    diagnostics_only = bool(diag_cfg.get("diagnostics_only", False))

    if diagnostics_enabled:
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

        if diagnostics_only:
            print("\nDiagnostics-only mode enabled. Skipping full sample/subsample estimation.")
            print(f"Summary statistics CSV: {summary_csv}")
            print(f"Summary statistics TEX: {summary_tex}")
            print("\nRun complete.")
            return

    layer3_cfg = config.get("layer3_diagnostics", {})
    layer3_enabled = bool(layer3_cfg.get("enabled", False))
    layer3_diagnostics_only = bool(layer3_cfg.get("diagnostics_only", False))

    if layer3_enabled:
        print("\n========================================")
        print("LAYER 3 GRID / BOUND DIAGNOSTICS")
        print("========================================")
        print(f"Diagnostics only mode: {layer3_diagnostics_only}")
        print(f"Fixed theta: {layer3_cfg.get('fixed_theta', config['theta0'])}")

        run_layer3_fixed_theta_diagnostics(
            df=df,
            config=config,
        )

        if layer3_diagnostics_only:
            print("\nLayer 3 diagnostics-only mode enabled. Skipping full sample/subsample estimation.")
            print(f"Summary statistics CSV: {summary_csv}")
            print(f"Summary statistics TEX: {summary_tex}")
            print("\nRun complete.")
            return

    full_results = run_single_estimation(
        df=df,
        config=config,
        sample_label="full_sample",
        theta0=theta0,
        save_outputs=True,
        make_plots=True,
        run_baseline_check=True,
        compute_standard_errors=True,
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
        run_baseline_check=True,
        compute_standard_errors=True,
    )

    large_results = run_single_estimation(
        df=large_df,
        config=config,
        sample_label="large_firms",
        theta0=theta0,
        save_outputs=True,
        make_plots=False,
        run_baseline_check=True,
        compute_standard_errors=True,
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
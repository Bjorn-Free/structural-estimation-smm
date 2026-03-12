from pathlib import Path
import copy
import sys
import time
from datetime import datetime

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
LOG_DIR = Path("results/logs")


class Tee:
    """
    Duplicate writes to multiple streams.
    Useful for saving terminal output to a log file while still printing live.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}h {minutes:02d}m {secs:05.2f}s"


def print_section(title):
    print("\n========================================")
    print(title)
    print("========================================")


def print_step(title):
    print("\n----------------------------------------")
    print(title)
    print("----------------------------------------")


def setup_logging(config):
    """
    Optionally duplicate stdout/stderr to a log file.
    Returns:
        log_file_handle, original_stdout, original_stderr
    """
    save_terminal_output = bool(config.get("save_terminal_output", False))
    terminal_output_path = config.get(
        "terminal_output_path",
        f"results/logs/runSMM_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    if not save_terminal_output:
        return None, None, None

    log_path = Path(terminal_output_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    print_section("LOGGING ENABLED")
    print(f"Timestamp: {timestamp_now()}")
    print(f"Terminal output is also being saved to: {log_path}")

    return log_file, original_stdout, original_stderr


def teardown_logging(log_file, original_stdout, original_stderr):
    """
    Restore stdout/stderr if logging was enabled.
    """
    if log_file is None:
        return

    sys.stdout.flush()
    sys.stderr.flush()

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    log_file.close()


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
    Starting point for the derivative-free SMM run.
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


def build_size_tercile_subsamples(df):
    """
    Build bottom-tercile and top-tercile size subsamples using total assets.

    The cleaned Compustat file uses the standard Compustat name:
        at = total assets

    Important:
    This is an observation-level split by firm-year total assets, not a
    permanent firm-level tercile classification.
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
    sample_start = time.perf_counter()

    moment_labels = moment_names(config)
    m_data = compute_moments(df, config)

    config_run = copy.deepcopy(config)
    config_run["theta0"] = np.asarray(theta0, dtype=float).tolist()

    print_section(f"RUNNING SAMPLE: {sample_label}")
    print(f"Timestamp: {timestamp_now()}")
    print(f"Sample observations: {len(df)}")

    print_vector_with_labels(
        title=f"{sample_label} empirical target moments",
        labels=moment_labels,
        values=m_data,
    )

    fixed_params_run = get_fixed_params(config_run, theta=theta0)

    print_step(f"{sample_label} baseline DP check")
    baseline_start = time.perf_counter()

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
    baseline_elapsed = time.perf_counter() - baseline_start
    print(f"Baseline block elapsed time: {format_seconds(baseline_elapsed)}")

    print_step(f"{sample_label} weighting matrix")
    weighting_start = time.perf_counter()

    W, weighting_details = make_weighting_matrix(
        df=df,
        config=config_run,
        return_details=True,
    )

    print("\nWeighting matrix constructed.")
    print(f"  ridge_scale              {weighting_details['ridge_scale']:.8e}")
    print(f"  condition_number         {weighting_details['condition_number']:.6e}")
    print(f"  used_pinv                {weighting_details['used_pinv']}")

    weighting_elapsed = time.perf_counter() - weighting_start
    print(f"Weighting matrix elapsed time: {format_seconds(weighting_elapsed)}")

    bounds = config_run["bounds"]

    print_step(f"{sample_label} SMM estimation")
    estimation_start = time.perf_counter()

    result = estimate_smm(
        theta0=np.asarray(theta0, dtype=float),
        bounds=bounds,
        m_data=m_data,
        W=W,
        config=config_run,
    )

    estimation_elapsed = time.perf_counter() - estimation_start
    print(f"SMM estimation elapsed time: {format_seconds(estimation_elapsed)}")

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

    print_step(f"{sample_label} standard errors")
    se_start = time.perf_counter()

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

    se_elapsed = time.perf_counter() - se_start
    print(f"Standard errors elapsed time: {format_seconds(se_elapsed)}")

    prefix = sample_label.lower().replace(" ", "_")
    saved_paths = {}

    if save_outputs:
        print_step(f"{sample_label} save outputs")
        output_start = time.perf_counter()

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

        _, estimation_settings_csv, estimation_settings_tex = (
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

        _, identification_detail_csv, identification_detail_tex = (
            save_identification_detailed_table(
                moment_labels=moment_labels,
                param_names=["psi", "lambda_external_finance", "rho", "sigma"],
                jacobian=se_results["jacobian"],
                decimals=6,
                filename_stem=f"{prefix}_identification_jacobian",
            )
        )

        _, identification_summary_csv, identification_summary_tex = (
            save_identification_summary_table(
                moment_labels=moment_labels,
                param_names=["psi", "lambda_external_finance", "rho", "sigma"],
                jacobian=se_results["jacobian"],
                bread_condition_number=se_results["bread_condition_number"],
                decimals=6,
                filename_stem=f"{prefix}_identification_summary",
            )
        )

        _, overid_note_csv, overid_note_tex = save_overidentification_note_table(
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

        output_elapsed = time.perf_counter() - output_start
        print(f"Output-saving elapsed time: {format_seconds(output_elapsed)}")
    else:
        param_table = pd.DataFrame()
        moment_table = pd.DataFrame()
        pretty_param_df = pd.DataFrame()
        estimated_policy_summary_df = pd.DataFrame()

    sample_elapsed = time.perf_counter() - sample_start

    print_section(f"FINISHED SAMPLE: {sample_label}")
    print(f"Timestamp: {timestamp_now()}")
    print(f"Total sample elapsed time: {format_seconds(sample_elapsed)}")

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
        "elapsed_seconds": sample_elapsed,
    }


def main():
    run_start = time.perf_counter()

    config = load_config("settings.json")
    log_file, original_stdout, original_stderr = setup_logging(config)

    try:
        debug_mode = bool(config.get("debug_mode", False))
        validation_mode = config.get("validation_mode", "fast")

        run_full_sample = bool(config.get("run_full_sample", True))
        run_small_firms = bool(config.get("run_small_firms", True))
        run_large_firms = bool(config.get("run_large_firms", True))

        print_section("RUN START")
        print(f"Timestamp: {timestamp_now()}")

        print_section("CONFIG LOAD CHECK")
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
        print(f"Run full sample: {run_full_sample}")
        print(f"Run small firms: {run_small_firms}")
        print(f"Run large firms: {run_large_firms}")

        if config.get("rebuild_data", False):
            print_step("REBUILD CLEAN DATA")
            rebuild_start = time.perf_counter()

            build_compustat(
                raw_path=config["raw_data_path"],
                clean_path=config["clean_data_path"],
                config=config,
            )

            rebuild_elapsed = time.perf_counter() - rebuild_start
            print(f"Rebuild-data elapsed time: {format_seconds(rebuild_elapsed)}")

        print_step("LOAD CLEAN DATA")
        load_start = time.perf_counter()
        df = load_clean_data(config["clean_data_path"])
        load_elapsed = time.perf_counter() - load_start
        print(f"Loaded clean data with {len(df)} observations.")
        print(f"Load-data elapsed time: {format_seconds(load_elapsed)}")

        print_step("SAVE SUMMARY STATISTICS")
        summary_start = time.perf_counter()
        summary_df, summary_csv, summary_tex = save_summary_statistics_table(
            df=df,
            decimals=4,
        )
        summary_elapsed = time.perf_counter() - summary_start
        print(f"Summary-statistics elapsed time: {format_seconds(summary_elapsed)}")

        spec = estimation_spec(config)
        theta0 = np.asarray(spec["theta0"], dtype=float)

        print_section("FULL SMM ESTIMATION RUN")
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

        full_results = None
        small_results = None
        large_results = None
        subsample_comparison_csv = None
        subsample_comparison_tex = None

        small_df = None
        large_df = None
        split_info = None

        if run_full_sample:
            print(f"\n[{timestamp_now()}] Starting full_sample estimation...")
            full_results = run_single_estimation(
                df=df,
                config=config,
                sample_label="full_sample",
                theta0=theta0,
                save_outputs=True,
                make_plots=True,
            )
            print(f"[{timestamp_now()}] Finished full_sample estimation.")
            print(f"Elapsed time: {format_seconds(full_results['elapsed_seconds'])}")

        if run_small_firms or run_large_firms:
            print_step("BUILD SIZE TERCILE SUBSAMPLES")
            split_start = time.perf_counter()

            small_df, large_df, split_info = build_size_tercile_subsamples(df)

            print_section("SIZE TERCILE SUBSAMPLES")
            print(f"Size variable: {split_info['size_variable']}")
            print(f"33rd percentile: {split_info['q33']:.6f}")
            print(f"67th percentile: {split_info['q67']:.6f}")
            print(f"Small-firm observations: {split_info['n_small']}")
            print(f"Large-firm observations: {split_info['n_large']}")

            split_elapsed = time.perf_counter() - split_start
            print(f"Subsample-build elapsed time: {format_seconds(split_elapsed)}")

        if run_small_firms:
            print(f"\n[{timestamp_now()}] Starting small_firms estimation...")
            small_results = run_single_estimation(
                df=small_df,
                config=config,
                sample_label="small_firms",
                theta0=theta0,
                save_outputs=True,
                make_plots=False,
            )
            print(f"[{timestamp_now()}] Finished small_firms estimation.")
            print(f"Elapsed time: {format_seconds(small_results['elapsed_seconds'])}")

        if run_large_firms:
            print(f"\n[{timestamp_now()}] Starting large_firms estimation...")
            large_results = run_single_estimation(
                df=large_df,
                config=config,
                sample_label="large_firms",
                theta0=theta0,
                save_outputs=True,
                make_plots=False,
            )
            print(f"[{timestamp_now()}] Finished large_firms estimation.")
            print(f"Elapsed time: {format_seconds(large_results['elapsed_seconds'])}")

        if (
            run_full_sample
            and run_small_firms
            and run_large_firms
            and full_results is not None
            and small_results is not None
            and large_results is not None
        ):
            print_step("SAVE SUBSAMPLE COMPARISON TABLE")
            subsample_start = time.perf_counter()

            subsample_comparison_df, subsample_comparison_csv, subsample_comparison_tex = (
                save_subsample_comparison_table(
                    full_sample=full_results,
                    small_firms=small_results,
                    large_firms=large_results,
                    split_info=split_info,
                    decimals=4,
                )
            )

            print_section("SUBSAMPLE PARAMETER COMPARISON TABLE")
            print(subsample_comparison_df.to_string(index=False))

            subsample_elapsed = time.perf_counter() - subsample_start
            print(f"Subsample-comparison elapsed time: {format_seconds(subsample_elapsed)}")
        else:
            subsample_comparison_df = None

        print_section("SUMMARY STATISTICS TABLE")
        print(summary_df.to_string(index=False))

        print_section("FILES SAVED")
        print(f"Summary statistics CSV: {summary_csv}")
        print(f"Summary statistics TEX: {summary_tex}")

        results_to_print = []
        if full_results is not None:
            results_to_print.append(("Full sample", full_results))
        if small_results is not None:
            results_to_print.append(("Small firms", small_results))
        if large_results is not None:
            results_to_print.append(("Large firms", large_results))

        for label, results_dict in results_to_print:
            print(f"\n{label}")
            for key, value in results_dict["saved_paths"].items():
                print(f"  {key}: {value}")

        if subsample_comparison_csv is not None:
            print(f"\nSubsample comparison CSV: {subsample_comparison_csv}")
            print(f"Subsample comparison TEX: {subsample_comparison_tex}")

        total_elapsed = time.perf_counter() - run_start

        print_section("RUN COMPLETE")
        print(f"End timestamp: {timestamp_now()}")
        print(f"Total elapsed time: {format_seconds(total_elapsed)}")

    finally:
        teardown_logging(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
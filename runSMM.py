import numpy as np
import pandas as pd

from src.config import load_config
from src.data import build_compustat, load_clean_data
from src.dp_solver import solve_investment_dp
from src.model import get_fixed_params
from src.moments import compute_moments, moment_names
from src.reporting import save_summary_statistics_table
from src.simulate import simulate_moments, simulate_panel
from src.diagnostics_plots import (
    plot_investment_policy_heatmap,
    plot_capital_distribution,
)


def build_policy_summary_table(solution, psi_value):

    k_grid = solution["k_grid"]
    z_grid = solution["z_grid"]
    policy_i = solution["policy_investment"]
    policy_kprime = solution["policy_kprime"]

    k_indices = [0, len(k_grid) // 2, len(k_grid) - 1]
    z_indices = [0, len(z_grid) // 2, len(z_grid) - 1]

    rows = []

    for ik in k_indices:
        for iz in z_indices:
            rows.append(
                {
                    "psi": float(psi_value),
                    "k_index": int(ik),
                    "z_index": int(iz),
                    "k": float(k_grid[ik]),
                    "z": float(z_grid[iz]),
                    "investment_policy": float(policy_i[ik, iz]),
                    "kprime_policy": float(policy_kprime[ik, iz]),
                }
            )

    return pd.DataFrame(rows)


def main():

    config = load_config("settings.json")
    debug = config.get("debug_mode", False)

    if config.get("rebuild_data", False):
        build_compustat(
            config["raw_data_path"],
            config["clean_data_path"],
            config,
        )

    df = load_clean_data(config["clean_data_path"])
    fixed = get_fixed_params(config)

    m_data = compute_moments(df, config)
    labels = moment_names(config)

    summary_df, summary_csv, summary_tex = save_summary_statistics_table(
        df=df,
        decimals=4,
    )

    # --------------------------------------------------
    # Validation scenarios
    # --------------------------------------------------

    if debug:
        print("\nDEBUG MODE ACTIVE — running minimal validation\n")

        validation_thetas = [
            {"label": "psi_debug", "theta": np.array(config["theta0"], dtype=float)}
        ]

    else:
        validation_thetas = [
            {"label": "psi_0p05", "theta": np.array([0.05, 0.05], dtype=float)},
            {"label": "psi_0p10", "theta": np.array([0.10, 0.05], dtype=float)},
            {"label": "psi_0p50", "theta": np.array([0.50, 0.05], dtype=float)},
            {"label": "psi_1p00", "theta": np.array([1.00, 0.05], dtype=float)},
        ]

    scenario_results = []
    policy_tables = []

    for scenario in validation_thetas:

        label = scenario["label"]
        theta = scenario["theta"]
        psi_value = float(theta[0])

        # --------------------------------------------------
        # Solve dynamic program
        # --------------------------------------------------

        solution = solve_investment_dp(
            theta=theta,
            config=config,
            fixed_params=fixed,
        )

        # --------------------------------------------------
        # Generate policy heatmap
        # --------------------------------------------------

        plot_investment_policy_heatmap(solution)

        # --------------------------------------------------
        # Simulate panel and capital distribution
        # --------------------------------------------------

        sim_df = simulate_panel(theta, config)
        plot_capital_distribution(sim_df)

        # --------------------------------------------------
        # Simulate moments
        # --------------------------------------------------

        m_sim = simulate_moments(theta, config)

        scenario_results.append(
            {
                "label": label,
                "theta": theta,
                "psi": psi_value,
                "solution": solution,
                "m_sim": np.asarray(m_sim, dtype=float),
            }
        )

        policy_tables.append(
            build_policy_summary_table(
                solution=solution,
                psi_value=psi_value,
            )
        )

    policy_summary_df = pd.concat(policy_tables, axis=0, ignore_index=True)

    print("Moment names:", labels)
    print("Data moments:", m_data)

    for result in scenario_results:

        solution = result["solution"]

        print("\nScenario:", result["label"])
        print("Theta:", result["theta"])
        print("Solver converged:", solution["converged"])
        print("Iterations:", solution["n_iter"])
        print("Bellman sup norm:", solution["max_diff"])
        print("Share lower bound:", solution["share_lower_bound"])
        print("Share upper bound:", solution["share_upper_bound"])
        print("Simulated moments:", result["m_sim"])

    print("\nPolicy summary:")
    print(policy_summary_df.to_string(index=False))

    print("\nSummary statistics table:")
    print(summary_df.to_string(index=False))

    print("\nSaved tables:")
    print(summary_csv)
    print(summary_tex)


if __name__ == "__main__":
    main()
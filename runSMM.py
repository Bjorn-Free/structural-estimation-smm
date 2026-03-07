import numpy as np
import pandas as pd

from src.config import load_config
from src.data import build_compustat, load_clean_data
from src.dp_solver import solve_investment_dp
from src.model import get_fixed_params
from src.moments import compute_moments, moment_names
from src.reporting import save_summary_statistics_table
from src.simulate import simulate_moments


def build_policy_summary_table(solution, psi_value):
    """
    Summarize the solved investment policy function at a few representative
    grid points for quick inspection.
    """
    k_grid = solution["k_grid"]
    z_grid = solution["z_grid"]
    policy_i = solution["policy_investment"]

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
                }
            )

    return pd.DataFrame(rows)


def build_moment_comparison_table(labels, m_data, scenario_results):
    """
    Build a table comparing data moments with simulated moments across
    fixed-parameter DP validation scenarios.
    """
    out = pd.DataFrame({"moment": labels, "data": np.asarray(m_data, dtype=float)})

    for result in scenario_results:
        scenario_label = result["label"]
        out[f"sim_{scenario_label}"] = np.asarray(result["m_sim"], dtype=float)
        out[f"gap_{scenario_label}"] = out["data"] - out[f"sim_{scenario_label}"]

    return out


def main():
    config = load_config("settings.json")

    if config.get("rebuild_data", False):
        build_compustat(
            config["raw_data_path"],
            config["clean_data_path"],
            config,
        )

    df = load_clean_data(config["clean_data_path"])
    fixed = get_fixed_params(config)

    # --------------------------------------------------
    # Data moments
    # --------------------------------------------------
    m_data = compute_moments(df, config)
    labels = moment_names(config)

    # --------------------------------------------------
    # Save summary statistics table
    # --------------------------------------------------
    summary_df, summary_csv, summary_tex = save_summary_statistics_table(
        df=df,
        decimals=4,
    )

    # --------------------------------------------------
    # DP validation scenarios
    # Keep lambda fixed for now because lambda is not yet inside the DP solver.
    # --------------------------------------------------
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

        solution = solve_investment_dp(
            theta=theta,
            config=config,
            fixed_params=fixed,
        )

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
    moment_validation_df = build_moment_comparison_table(
        labels=labels,
        m_data=m_data,
        scenario_results=scenario_results,
    )

    # --------------------------------------------------
    # Console output
    # --------------------------------------------------
    print("Moment names:", labels)
    print("Data moments:", m_data)

    print("\nDynamic programming validation results:")
    for result in scenario_results:
        solution = result["solution"]
        print(f"\nScenario: {result['label']}")
        print("Theta:", result["theta"])
        print("Solver converged:", solution["converged"])
        print("Iterations:", solution["n_iter"])
        print("Final Bellman sup norm:", solution["max_diff"])
        print("Simulated moments:", result["m_sim"])

    print("\nPolicy function summary table:")
    print(policy_summary_df.to_string(index=False))

    print("\nMoment validation table:")
    print(moment_validation_df.to_string(index=False))

    print("\nSummary statistics table:")
    print(summary_df.to_string(index=False))

    print("\nSaved tables:")
    print(f"- {summary_csv}")
    print(f"- {summary_tex}")


if __name__ == "__main__":
    main()
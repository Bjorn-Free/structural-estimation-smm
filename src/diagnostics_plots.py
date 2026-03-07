import os
import numpy as np
import matplotlib.pyplot as plt


def plot_investment_policy_heatmap(solution, save_path="results/figures"):
    """
    Plot heatmap of the solved investment policy i(k,z).

    Parameters
    ----------
    solution : dict
        Output of solve_investment_dp()

    save_path : str
        Directory where figure will be saved
    """
    os.makedirs(save_path, exist_ok=True)

    k_grid = solution["k_grid"]
    z_grid = solution["z_grid"]
    policy_i = solution["policy_investment"]

    plt.figure(figsize=(8, 6))

    heatmap = plt.contourf(
        z_grid,
        k_grid,
        policy_i,
        levels=20,
    )

    plt.colorbar(heatmap, label="Investment rate (i)")
    plt.xlabel("Productivity (z)")
    plt.ylabel("Capital (k)")
    plt.title("Optimal Investment Policy Function")

    file_path = os.path.join(save_path, "investment_policy_heatmap.png")

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"Saved investment policy heatmap: {file_path}")


def plot_capital_distribution(sim_df, save_path="results/figures"):
    """
    Plot the distribution of simulated capital.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulated panel data. Must contain a 'capital' column.

    save_path : str
        Directory where figure will be saved
    """
    os.makedirs(save_path, exist_ok=True)

    if "capital" not in sim_df.columns:
        raise KeyError(
            "Simulated data frame must contain a 'capital' column "
            "to plot the capital distribution."
        )

    capital = sim_df["capital"].to_numpy(dtype=float)
    capital = capital[np.isfinite(capital)]
    capital = capital[capital > 0.0]

    if capital.size == 0:
        raise ValueError("No positive finite capital observations found in sim_df.")

    plt.figure(figsize=(8, 6))
    plt.hist(capital, bins=30)

    plt.xlabel("Capital")
    plt.ylabel("Frequency")
    plt.title("Distribution of Simulated Capital")

    file_path = os.path.join(save_path, "capital_distribution.png")

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"Saved capital distribution plot: {file_path}")
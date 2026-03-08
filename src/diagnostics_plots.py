import os
import numpy as np
import matplotlib.pyplot as plt


def plot_investment_policy_heatmap(solution, save_path="results/figures"):
    """
    Plot a heatmap of the solved investment policy using the middle cash slice.

    Intended for the structural cash stage where the solution contains:
    - k_grid
    - p_grid
    - z_grid
    - policy_investment with shape (nk, np, nz)

    We use the midpoint of the cash grid to keep the diagnostic simple.
    """
    os.makedirs(save_path, exist_ok=True)

    k_grid = solution["k_grid"]
    p_grid = solution["p_grid"]
    z_grid = solution["z_grid"]
    policy_i = solution["policy_investment"]

    p_index = len(p_grid) // 2
    p_value = float(p_grid[p_index])

    policy_slice = policy_i[:, p_index, :]

    plt.figure(figsize=(8, 6))

    heatmap = plt.contourf(
        z_grid,
        k_grid,
        policy_slice,
        levels=20,
    )

    plt.colorbar(heatmap, label="Investment rate (i)")
    plt.xlabel("Productivity (z)")
    plt.ylabel("Capital (k)")
    plt.title(f"Optimal Investment Policy Function (cash slice = {p_value:.2f})")

    file_path = os.path.join(save_path, "investment_policy_heatmap.png")

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"Saved investment policy heatmap: {file_path}")


def plot_capital_distribution(sim_df, save_path="results/figures"):
    """
    Plot the distribution of simulated capital.
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
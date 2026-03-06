import numpy as np
import pandas as pd

from src.moments import moment_names
from src.simulate import simulate_moments


def run_parameter_diagnostics(config: dict, theta_list: list, labels: list[str] | None = None) -> pd.DataFrame:
    """
    Simulate model moments for a list of parameter vectors and return
    a clean comparison table.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.
    theta_list : list
        List of parameter vectors, each of the form [psi, lambda].
    labels : list[str] | None
        Optional labels for each parameter vector.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per scenario and one column per moment.
    """
    if labels is None:
        labels = [f"scenario_{i+1}" for i in range(len(theta_list))]

    if len(labels) != len(theta_list):
        raise ValueError("labels and theta_list must have the same length.")

    names = moment_names(config)
    rows = []

    for label, theta in zip(labels, theta_list):
        theta = np.asarray(theta, dtype=float)
        m_sim = simulate_moments(theta, config)

        row = {
            "scenario": label,
            "psi": float(theta[0]),
            "lambda": float(theta[1]),
        }

        for name, value in zip(names, m_sim):
            row[name] = float(value)

        rows.append(row)

    df_diag = pd.DataFrame(rows)
    return df_diag


def print_parameter_diagnostics(df_diag: pd.DataFrame, decimals: int = 4) -> None:
    """
    Print a compact diagnostics table in the terminal.
    """
    if df_diag.empty:
        print("Diagnostics table is empty.")
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\nParameter diagnostics table:")
    print(df_diag.round(decimals).to_string(index=False))
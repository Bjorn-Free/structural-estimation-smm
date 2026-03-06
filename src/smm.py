import numpy as np


def make_weighting_matrix(data_df, config: dict) -> np.ndarray:
    """
    Placeholder weighting matrix for SMM.
    For now: identity matrix.
    """
    n_moments = int(config.get("n_moments", 2))
    return np.eye(n_moments)


def estimate_smm(theta0, bounds, data_mom: np.ndarray, W: np.ndarray, config: dict) -> dict:
    """
    Placeholder SMM estimator.
    For now: just returns the starting values.
    """
    theta_hat = np.array(theta0, dtype=float)

    return {
        "theta_hat": theta_hat,
        "objective_value": None,
        "converged": True
    }
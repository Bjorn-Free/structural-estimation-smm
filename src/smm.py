import numpy as np
from scipy.optimize import minimize

from src.simulate import simulate_moments


def make_weighting_matrix(df, config):
    """
    Create the SMM weighting matrix.

    For now, use the identity matrix.
    """
    n_moments = int(config["n_moments"])
    return np.eye(n_moments)


def smm_objective(theta, m_data, W, config):
    """
    SMM objective:
        Q(theta) = (m_data - m_sim(theta))' W (m_data - m_sim(theta))
    """
    m_sim = simulate_moments(theta, config)
    diff = np.asarray(m_data, dtype=float) - np.asarray(m_sim, dtype=float)
    obj = float(diff.T @ W @ diff)
    return obj


def estimate_smm(theta0, bounds, m_data, W, config):
    """
    Estimate structural parameters by minimizing the SMM objective.
    """
    theta0 = np.asarray(theta0, dtype=float)
    bounds = [tuple(b) for b in bounds]

    result = minimize(
        fun=smm_objective,
        x0=theta0,
        args=(m_data, W, config),
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_hat = np.asarray(result.x, dtype=float)
    m_sim_hat = simulate_moments(theta_hat, config)
    diff_hat = np.asarray(m_data, dtype=float) - np.asarray(m_sim_hat, dtype=float)
    objective_value = float(diff_hat.T @ W @ diff_hat)

    return {
        "theta_hat": theta_hat,
        "objective_value": objective_value,
        "m_data": np.asarray(m_data, dtype=float),
        "m_sim": np.asarray(m_sim_hat, dtype=float),
        "success": result.success,
        "message": result.message,
        "n_iter": result.nit,
        "optimizer_result": result,
    }
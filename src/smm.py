import numpy as np
from scipy.optimize import minimize

from src.moments import moment_covariance_matrix
from src.simulate import simulate_moments


def make_weighting_matrix(df, config, return_details=False):
    """
    Construct the efficient SMM weighting matrix.

    Course-aligned target:
        W = S^{-1}

    where S is the estimated covariance matrix of the empirical
    sample moment vector, computed from the real data.

    Practical numerical details:
    1. Force S to be exactly symmetric.
    2. Add a small ridge term scaled to the magnitude of S.
    3. Use a pseudo-inverse fallback if direct inversion fails.
    """
    S = np.asarray(moment_covariance_matrix(df, config), dtype=float)

    n_moments = int(config["n_moments"])
    if S.shape != (n_moments, n_moments):
        raise ValueError(
            f"Moment covariance matrix has incorrect shape {S.shape}; "
            f"expected ({n_moments}, {n_moments})."
        )

    if not np.isfinite(S).all():
        raise ValueError("Moment covariance matrix contains non-finite values.")

    # Force exact symmetry
    S = 0.5 * (S + S.T)

    diag_S = np.diag(S).copy()
    avg_diag = float(np.mean(diag_S)) if diag_S.size > 0 else 1.0

    # Scaled ridge for finite-sample numerical stability
    ridge_scale = max(1e-8, 1e-6 * max(abs(avg_diag), 1.0))
    ridge = ridge_scale * np.eye(S.shape[0])

    S_stable = S + ridge

    try:
        cond_number = float(np.linalg.cond(S_stable))
    except np.linalg.LinAlgError:
        cond_number = np.inf

    used_pinv = False
    try:
        W = np.linalg.inv(S_stable)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(S_stable)
        used_pinv = True

    W = 0.5 * (W + W.T)

    if not np.isfinite(W).all():
        raise ValueError("Weighting matrix contains non-finite values.")

    details = {
        "S": S,
        "S_stable": S_stable,
        "diag_S": diag_S,
        "ridge_scale": ridge_scale,
        "condition_number": cond_number,
        "used_pinv": used_pinv,
    }

    if return_details:
        return W, details

    return W


def identity_weighting_matrix(config):
    """
    Construct the identity weighting matrix for baseline comparison.
    """
    n_moments = int(config["n_moments"])
    return np.eye(n_moments, dtype=float)


def smm_objective(theta, m_data, W, config):
    """
    SMM objective function:

        Q(theta) = (m_data - m_sim(theta))' W (m_data - m_sim(theta))
    """
    theta = np.asarray(theta, dtype=float)
    m_data = np.asarray(m_data, dtype=float)
    W = np.asarray(W, dtype=float)

    m_sim = np.asarray(simulate_moments(theta, config), dtype=float)

    if m_sim.shape != m_data.shape:
        raise ValueError(
            f"Simulated moments shape {m_sim.shape} does not match "
            f"data moments shape {m_data.shape}."
        )

    diff = m_data - m_sim
    obj = float(diff.T @ W @ diff)

    if not np.isfinite(obj):
        raise ValueError("SMM objective evaluated to a non-finite value.")

    return obj


def objective_contribution_matrix(moment_errors, W):
    """
    Build the matrix of pairwise objective contributions:

        C = diag(e) @ W @ diag(e)

    so that:
        Q = sum_{i,j} C[i,j]

    The diagonal elements C[i,i] show the direct own-moment contributions.
    The row sums show each moment's total contribution including covariance
    interactions with other moments.
    """
    e = np.asarray(moment_errors, dtype=float)
    W = np.asarray(W, dtype=float)

    if e.ndim != 1:
        raise ValueError("moment_errors must be a 1D vector.")

    if W.shape != (e.size, e.size):
        raise ValueError(
            f"W has shape {W.shape}, but moment_errors has length {e.size}."
        )

    D = np.diag(e)
    C = D @ W @ D

    if not np.isfinite(C).all():
        raise ValueError("Objective contribution matrix contains non-finite values.")

    return C


def estimate_smm(theta0, bounds, m_data, W, config):
    """
    Estimate structural parameters by minimizing the SMM objective.
    """
    theta0 = np.asarray(theta0, dtype=float)
    m_data = np.asarray(m_data, dtype=float)
    W = np.asarray(W, dtype=float)
    bounds = [tuple(b) for b in bounds]

    result = minimize(
        fun=smm_objective,
        x0=theta0,
        args=(m_data, W, config),
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_hat = np.asarray(result.x, dtype=float)
    m_sim_hat = np.asarray(simulate_moments(theta_hat, config), dtype=float)

    if m_sim_hat.shape != m_data.shape:
        raise ValueError(
            f"Simulated moments at theta_hat have shape {m_sim_hat.shape}, "
            f"but data moments have shape {m_data.shape}."
        )

    diff_hat = m_data - m_sim_hat
    objective_value = float(diff_hat.T @ W @ diff_hat)

    if not np.isfinite(objective_value):
        raise ValueError("Final SMM objective value is non-finite.")

    contribution_matrix = objective_contribution_matrix(diff_hat, W)
    own_contributions = np.diag(contribution_matrix)
    total_contributions = contribution_matrix.sum(axis=1)

    return {
        "theta_hat": theta_hat,
        "objective_value": objective_value,
        "m_data": m_data,
        "m_sim": m_sim_hat,
        "moment_errors": diff_hat,
        "objective_contribution_matrix": contribution_matrix,
        "objective_own_contributions": own_contributions,
        "objective_total_contributions": total_contributions,
        "success": bool(result.success),
        "message": result.message,
        "n_iter": int(result.nit),
        "optimizer_result": result,
    }
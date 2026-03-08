import copy

import numpy as np
from scipy.optimize import minimize

from src.moments import moment_covariance_matrix
from src.simulate import simulate_moments
from src.dp_solver import solve_investment_dp
from src.model import get_fixed_params


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

    S = 0.5 * (S + S.T)

    diag_S = np.diag(S).copy()
    avg_diag = float(np.mean(diag_S)) if diag_S.size > 0 else 1.0

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


def _solve_dp_for_theta(theta, config):
    """
    Solve the DP once for a given theta/config pair.

    We deep-copy config so temporary caches or run-specific mutations do not
    pollute the outer estimation environment.
    """
    config_run = copy.deepcopy(config)
    fixed_params_run = get_fixed_params(config_run)

    solution = solve_investment_dp(
        theta=np.asarray(theta, dtype=float),
        config=config_run,
        fixed_params=fixed_params_run,
    )
    return solution, config_run


def _simulate_moments_from_theta(theta, config):
    """
    Solve the DP once, then simulate moments using the supplied policy functions.

    This is the key structural-estimation speed improvement:
    do not let the simulator solve the DP a second time.
    """
    solution, config_run = _solve_dp_for_theta(theta, config)

    m_sim = np.asarray(
        simulate_moments(theta, config_run, solution=solution),
        dtype=float,
    )

    return m_sim, solution, config_run


def smm_objective(theta, m_data, W, config):
    """
    SMM objective function:

        Q(theta) = (m_data - m_sim(theta))' W (m_data - m_sim(theta))

    Performance note
    ----------------
    This implementation solves the DP exactly once per objective evaluation and
    then reuses the solved policy functions inside simulation.
    """
    theta = np.asarray(theta, dtype=float)
    m_data = np.asarray(m_data, dtype=float)
    W = np.asarray(W, dtype=float)

    m_sim, _, _ = _simulate_moments_from_theta(theta, config)

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


def _normalize_theta_to_unit_box(theta, bounds):
    """
    Map theta from bounded parameter space to unit box coordinates.
    """
    theta = np.asarray(theta, dtype=float)
    x = np.empty_like(theta, dtype=float)

    for i, (value, bound) in enumerate(zip(theta, bounds)):
        lower, upper = float(bound[0]), float(bound[1])
        width = upper - lower

        if width <= 0.0:
            raise ValueError(f"Invalid bound width at index {i}: {bound}")

        x[i] = (value - lower) / width

    return np.clip(x, 0.0, 1.0)


def _map_unit_box_to_theta(x, bounds):
    """
    Map unit box coordinates back into bounded parameter space.
    """
    x = np.asarray(x, dtype=float)
    theta = np.empty_like(x, dtype=float)

    for i, (value, bound) in enumerate(zip(x, bounds)):
        lower, upper = float(bound[0]), float(bound[1])
        theta[i] = lower + np.clip(value, 0.0, 1.0) * (upper - lower)

    return theta


def _penalized_unit_box_objective(x, m_data, W, config, bounds):
    """
    Derivative-free objective on the unit box.

    We optimize over x in R^n, then:
    - clip x softly through a quadratic penalty if it leaves [0,1]^n
    - map back to theta using the original parameter bounds

    This lets us use Nelder-Mead while still respecting the original bounds.
    """
    x = np.asarray(x, dtype=float)

    below = np.minimum(x, 0.0)
    above = np.maximum(x - 1.0, 0.0)
    penalty = 1e6 * float(np.sum(below**2 + above**2))

    x_clipped = np.clip(x, 0.0, 1.0)
    theta = _map_unit_box_to_theta(x_clipped, bounds)

    return smm_objective(theta, m_data, W, config) + penalty


def _get_optimizer_method(config):
    """
    Retrieve the optimizer method for the current run.
    """
    return str(config.get("smm_optimizer_method", "Nelder-Mead"))


def _get_optimizer_options(config):
    """
    Retrieve and normalize optimizer options for the current capped run.
    """
    optimizer_options = copy.deepcopy(config.get("smm_optimizer_options", {}))

    if not optimizer_options:
        optimizer_options = {
            "maxiter": 40,
            "maxfev": 80,
            "xatol": 1e-3,
            "fatol": 1e-3,
            "adaptive": True,
        }

    optimizer_options.setdefault("maxiter", 40)
    optimizer_options.setdefault("maxfev", 80)
    optimizer_options.setdefault("xatol", 1e-3)
    optimizer_options.setdefault("fatol", 1e-3)
    optimizer_options.setdefault("adaptive", True)

    if "disp" in optimizer_options:
        optimizer_options.pop("disp")

    return optimizer_options


def estimate_smm(theta0, bounds, m_data, W, config):
    """
    Estimate structural parameters by minimizing the SMM objective.

    Current recommended capped test:
    - use a derivative-free optimizer
    - keep the run short
    - verify that the objective moves from the starting point

    Implementation detail
    ---------------------
    We optimize on the unit box using Nelder-Mead and map back to theta
    using the original parameter bounds.
    """
    theta0 = np.asarray(theta0, dtype=float)
    m_data = np.asarray(m_data, dtype=float)
    W = np.asarray(W, dtype=float)
    bounds = [tuple(b) for b in bounds]

    optimizer_method = _get_optimizer_method(config)
    optimizer_options = _get_optimizer_options(config)

    if optimizer_method != "Nelder-Mead":
        raise ValueError(
            "This capped derivative-free test is configured for "
            "smm_optimizer_method = 'Nelder-Mead'."
        )

    objective_initial = smm_objective(theta0, m_data, W, config)

    x0 = _normalize_theta_to_unit_box(theta0, bounds)

    result = minimize(
        fun=_penalized_unit_box_objective,
        x0=x0,
        args=(m_data, W, config, bounds),
        method=optimizer_method,
        options=optimizer_options,
    )

    theta_hat = _map_unit_box_to_theta(result.x, bounds)

    m_sim_hat, solution_hat, config_hat = _simulate_moments_from_theta(theta_hat, config)

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

    n_fun_eval = getattr(result, "nfev", np.nan)

    return {
        "theta_hat": np.asarray(theta_hat, dtype=float),
        "objective_initial": float(objective_initial),
        "objective_value": objective_value,
        "objective_improvement": float(objective_initial - objective_value),
        "m_data": m_data,
        "m_sim": m_sim_hat,
        "moment_errors": diff_hat,
        "objective_contribution_matrix": contribution_matrix,
        "objective_own_contributions": own_contributions,
        "objective_total_contributions": total_contributions,
        "success": bool(result.success),
        "message": result.message,
        "n_iter": int(getattr(result, "nit", 0)),
        "n_fun_eval": int(n_fun_eval) if np.isfinite(n_fun_eval) else -1,
        "optimizer_result": result,
        "solution_hat": solution_hat,
        "config_hat": config_hat,
        "optimizer_method": optimizer_method,
        "optimizer_options": optimizer_options,
    }
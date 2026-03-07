import math
import numpy as np


def get_dp_settings(config: dict) -> dict:
    """
    Read dynamic-programming settings from config.
    """
    dp_cfg = config.get("dp", {})

    return {
        "beta": float(dp_cfg.get("beta", 0.95)),
        "k_grid_size": int(dp_cfg.get("k_grid_size", 25)),
        "k_min": float(dp_cfg.get("k_min", 20.0)),
        "k_max": float(dp_cfg.get("k_max", 250.0)),
        "z_grid_size": int(dp_cfg.get("z_grid_size", 7)),
        "tauchen_m": float(dp_cfg.get("tauchen_m", 3.0)),
        "control_grid_size": int(dp_cfg.get("control_grid_size", 41)),
        "investment_min": float(dp_cfg.get("investment_min", -0.20)),
        "investment_max": float(dp_cfg.get("investment_max", 0.80)),
        "max_iter": int(dp_cfg.get("max_iter", 500)),
        "tol": float(dp_cfg.get("tol", 1e-6)),
    }


def _std_norm_cdf(x):
    """
    Standard normal CDF using erf.
    """
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


def tauchen(rho: float, sigma: float, n: int, m: float = 3.0):
    """
    Tauchen discretization for AR(1):

        x' = rho x + sigma * eps, eps ~ N(0,1)

    Returns
    -------
    grid : ndarray
        Discretized state grid for x.

    transition : ndarray
        Markov transition matrix of shape (n, n).
    """
    if n < 2:
        raise ValueError("Tauchen discretization requires n >= 2.")

    sigma_x = sigma / np.sqrt(1.0 - rho ** 2)
    x_max = m * sigma_x
    x_min = -x_max

    grid = np.linspace(x_min, x_max, n)
    step = grid[1] - grid[0]

    transition = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if j == 0:
                z_upper = (grid[j] - rho * grid[i] + step / 2.0) / sigma
                transition[i, j] = _std_norm_cdf(z_upper)
            elif j == n - 1:
                z_lower = (grid[j] - rho * grid[i] - step / 2.0) / sigma
                transition[i, j] = 1.0 - _std_norm_cdf(z_lower)
            else:
                z_upper = (grid[j] - rho * grid[i] + step / 2.0) / sigma
                z_lower = (grid[j] - rho * grid[i] - step / 2.0) / sigma
                transition[i, j] = _std_norm_cdf(z_upper) - _std_norm_cdf(z_lower)

    transition = transition / transition.sum(axis=1, keepdims=True)
    return grid, transition


def build_k_grid(config: dict) -> np.ndarray:
    """
    Capital grid.
    """
    dp = get_dp_settings(config)
    grid = np.linspace(dp["k_min"], dp["k_max"], dp["k_grid_size"])
    return grid.astype(float)


def build_investment_grid(config: dict) -> np.ndarray:
    """
    Control grid for the investment rate i.
    """
    dp = get_dp_settings(config)
    grid = np.linspace(
        dp["investment_min"],
        dp["investment_max"],
        dp["control_grid_size"],
    )
    return grid.astype(float)


def build_z_process(config: dict, fixed_params: dict):
    """
    Discretize the productivity process in logs and map to levels.
    """
    dp = get_dp_settings(config)
    rho = float(fixed_params["rho"])
    sigma = float(fixed_params["sigma"])

    log_z_grid, transition = tauchen(
        rho=rho,
        sigma=sigma,
        n=dp["z_grid_size"],
        m=dp["tauchen_m"],
    )

    z_grid = np.exp(log_z_grid)
    return log_z_grid.astype(float), z_grid.astype(float), transition.astype(float)


def profit_rate(z, alpha, profit_intercept):
    """
    Profitability rate per unit of capital.
    """
    z = np.maximum(z, 1e-8)
    return profit_intercept + (z ** alpha) - 1.0


def operating_profit(k, z, alpha, profit_intercept):
    """
    Operating profit level.
    """
    k = np.maximum(k, 1e-8)
    return k * profit_rate(z, alpha, profit_intercept)


def adjustment_cost(k, investment_rate, psi):
    """
    Convex adjustment cost:
        AC = 0.5 * psi * i^2 * k
    """
    return 0.5 * psi * (investment_rate ** 2) * k


def next_capital(k, investment_rate, delta):
    """
    Law of motion for capital:
        k' = (1 - delta + i) * k
    """
    return (1.0 - delta + investment_rate) * k


def one_period_payoff(k, z, investment_rate, psi, fixed_params):
    """
    One-period payoff for the minimal (k,z) investment model.

    Current simplified objective:
        payoff = operating profit - investment expenditure - adjustment cost

    This is the correct minimal starting point for debugging the dynamic solver.
    External finance, debt, and cash will be added in later stages.
    """
    alpha = float(fixed_params["alpha"])
    delta = float(fixed_params["delta"])
    profit_intercept = float(fixed_params["profit_intercept"])

    prof = operating_profit(
        k=k,
        z=z,
        alpha=alpha,
        profit_intercept=profit_intercept,
    )

    investment_expenditure = investment_rate * k
    adj_cost = adjustment_cost(
        k=k,
        investment_rate=investment_rate,
        psi=psi,
    )

    payoff = prof - investment_expenditure - adj_cost
    k_next = next_capital(k, investment_rate, delta)

    return float(payoff), float(k_next)


def linear_interp_1d(x_grid: np.ndarray, y_values: np.ndarray, x: float) -> float:
    """
    1D linear interpolation with flat extrapolation at the boundaries.
    """
    x = float(x)

    if x <= x_grid[0]:
        return float(y_values[0])
    if x >= x_grid[-1]:
        return float(y_values[-1])

    idx = np.searchsorted(x_grid, x) - 1
    idx = max(0, min(idx, len(x_grid) - 2))

    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]
    y0 = y_values[idx]
    y1 = y_values[idx + 1]

    weight = (x - x0) / (x1 - x0)
    return float((1.0 - weight) * y0 + weight * y1)


def expected_continuation_value(k_next, iz, V, k_grid, z_transition):
    """
    Compute E[V(k', z') | current z-index = iz] using interpolation over k.
    """
    probs = z_transition[iz]
    expected_value = 0.0

    for iz_next in range(len(probs)):
        V_next_z = V[:, iz_next]
        cont_value = linear_interp_1d(k_grid, V_next_z, k_next)
        expected_value += probs[iz_next] * cont_value

    return float(expected_value)


def solve_investment_dp(theta, config: dict, fixed_params: dict):
    """
    Solve the minimal dynamic investment model with states (k, z)
    and control i using value function iteration.
    """
    theta = np.asarray(theta, dtype=float)
    psi = float(theta[0])

    dp = get_dp_settings(config)
    beta = dp["beta"]

    k_grid = build_k_grid(config)
    log_z_grid, z_grid, z_transition = build_z_process(config, fixed_params)
    investment_grid = build_investment_grid(config)

    nk = len(k_grid)
    nz = len(z_grid)

    V = np.zeros((nk, nz), dtype=float)
    policy_i = np.zeros((nk, nz), dtype=float)

    converged = False
    max_diff = np.inf

    for it in range(dp["max_iter"]):
        V_new = np.zeros_like(V)

        for ik, k in enumerate(k_grid):
            for iz, z in enumerate(z_grid):
                best_value = -np.inf
                best_i = investment_grid[0]

                for investment_rate in investment_grid:
                    payoff, k_next = one_period_payoff(
                        k=k,
                        z=z,
                        investment_rate=investment_rate,
                        psi=psi,
                        fixed_params=fixed_params,
                    )

                    if k_next <= 1e-8:
                        continue

                    continuation = expected_continuation_value(
                        k_next=k_next,
                        iz=iz,
                        V=V,
                        k_grid=k_grid,
                        z_transition=z_transition,
                    )

                    candidate_value = payoff + beta * continuation

                    if candidate_value > best_value:
                        best_value = candidate_value
                        best_i = investment_rate

                V_new[ik, iz] = best_value
                policy_i[ik, iz] = best_i

        max_diff = float(np.max(np.abs(V_new - V)))
        V = V_new

        if max_diff < dp["tol"]:
            converged = True
            break

    return {
        "k_grid": k_grid,
        "log_z_grid": log_z_grid,
        "z_grid": z_grid,
        "z_transition": z_transition,
        "investment_grid": investment_grid,
        "value_function": V,
        "policy_investment": policy_i,
        "converged": converged,
        "n_iter": it + 1,
        "max_diff": max_diff,
    }


def interpolate_policy_investment(k, log_z, solution):
    """
    Evaluate the solved investment policy approximately at continuous states.

    Strategy:
    - nearest-neighbor in log z
    - linear interpolation in k
    """
    k_grid = solution["k_grid"]
    log_z_grid = solution["log_z_grid"]
    policy_i = solution["policy_investment"]

    iz = int(np.argmin(np.abs(log_z_grid - log_z)))
    policy_slice = policy_i[:, iz]

    return linear_interp_1d(k_grid, policy_slice, k)
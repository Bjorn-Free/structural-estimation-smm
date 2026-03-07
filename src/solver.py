import math
import numpy as np

from src.model import (
    profit,
    adjustment_cost,
    net_investment,
    external_finance_needed,
    external_finance_cost,
)


def get_dp_settings(config: dict) -> dict:
    """
    Read dynamic-programming settings from config.
    """
    dp_cfg = config.get("dp", {})

    return {
        "beta": float(dp_cfg.get("beta", 0.95)),
        "k_grid_size": int(dp_cfg.get("k_grid_size", 41)),
        "k_min": float(dp_cfg.get("k_min", 10.0)),
        "k_max": float(dp_cfg.get("k_max", 300.0)),
        "z_grid_size": int(dp_cfg.get("z_grid_size", 7)),
        "tauchen_m": float(dp_cfg.get("tauchen_m", 2.5)),
        "control_grid_size": int(dp_cfg.get("control_grid_size", 181)),
        "investment_min": float(dp_cfg.get("investment_min", -0.35)),
        "investment_max": float(dp_cfg.get("investment_max", 0.50)),
        "max_iter": int(dp_cfg.get("max_iter", 500)),
        "tol": float(dp_cfg.get("tol", 1e-6)),
        "policy_lower_bound_tolerance": float(
            dp_cfg.get("policy_lower_bound_tolerance", 1e-8)
        ),
        "policy_upper_bound_tolerance": float(
            dp_cfg.get("policy_upper_bound_tolerance", 1e-8)
        ),
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

    A log-spaced grid gives better resolution in the low-capital region.
    """
    dp = get_dp_settings(config)

    k_min = float(dp["k_min"])
    k_max = float(dp["k_max"])
    n = int(dp["k_grid_size"])

    grid = np.exp(np.linspace(np.log(k_min), np.log(k_max), n))
    return grid.astype(float)


def build_kprime_grid(config: dict, current_k: float, delta: float) -> np.ndarray:
    """
    Control grid for next-period capital k'.

    From:
        k' = (1 - delta + i) * k

    The feasible k' set is intersected with the support of the k-grid itself.
    """
    dp = get_dp_settings(config)

    i_min = float(dp["investment_min"])
    i_max = float(dp["investment_max"])
    k_min = float(dp["k_min"])
    k_max = float(dp["k_max"])
    n = int(dp["control_grid_size"])

    implied_kprime_min = max((1.0 - delta + i_min) * current_k, 1e-8)
    implied_kprime_max = max(
        (1.0 - delta + i_max) * current_k,
        implied_kprime_min + 1e-8,
    )

    kprime_min = max(implied_kprime_min, k_min)
    kprime_max = min(implied_kprime_max, k_max)

    if kprime_max <= kprime_min:
        kprime_max = kprime_min + 1e-8

    if kprime_min <= 0.0:
        kprime_min = 1e-8

    grid = np.exp(np.linspace(np.log(kprime_min), np.log(kprime_max), n))
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


def investment_rate_from_kprime(k: float, kprime: float, delta: float) -> float:
    """
    Recover the investment rate implied by choosing next-period capital:

        i = k'/k - (1 - delta)
    """
    k = max(float(k), 1e-8)
    kprime = max(float(kprime), 1e-8)
    return float((kprime / k) - (1.0 - delta))


def one_period_payoff_from_kprime(k, z, kprime, theta, fixed_params):
    """
    One-period payoff for the minimal canonical model.

    State:
        (k, z)

    Control:
        k'

    Period logic:
        1. operating profits are generated from the production side
        2. net investment expenditure is implied by k'
        3. adjustment costs depend on the investment rate
        4. if internal funds are insufficient, external finance is raised
        5. external finance incurs proportional cost lambda

    Current period payoff:
        payout = profits - investment - adjustment_cost - lambda * ext_finance

    where:
        ext_finance = max(investment + adjustment_cost - profits, 0)

    This is the current minimal Section 3.1 / 3.2 style backbone before debt
    and cash are introduced as explicit states.
    """
    theta = np.asarray(theta, dtype=float)
    psi = float(theta[0])
    lam = float(theta[1])

    alpha = float(fixed_params["alpha"])
    delta = float(fixed_params["delta"])
    production_scale = float(fixed_params["production_scale"])
    fixed_cost = float(fixed_params["fixed_cost"])
    profit_intercept = float(fixed_params["profit_intercept"])

    i_rate = investment_rate_from_kprime(k=k, kprime=kprime, delta=delta)

    prof = profit(
        k=k,
        z=z,
        alpha=alpha,
        profit_intercept=profit_intercept,
        production_scale=production_scale,
        fixed_cost=fixed_cost,
    )

    investment = net_investment(
        k=k,
        kprime=kprime,
        delta=delta,
    )

    adj_cost = adjustment_cost(
        k=k,
        investment_rate=i_rate,
        psi=psi,
    )

    ext_finance = external_finance_needed(
        profits=prof,
        investment=investment,
        adj_cost=adj_cost,
        debt_change=0.0,
        cash_change=0.0,
    )

    ext_fin_cost = external_finance_cost(
        ext_finance=ext_finance,
        lam=lam,
    )

    payoff = prof - investment - adj_cost - ext_fin_cost

    return {
        "payoff": float(payoff),
        "investment_rate": float(i_rate),
        "investment": float(investment),
        "profit": float(prof),
        "adjustment_cost": float(adj_cost),
        "external_finance": float(ext_finance),
        "external_finance_cost": float(ext_fin_cost),
    }


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


def compute_policy_bound_shares(policy_i: np.ndarray, config: dict) -> dict:
    """
    Compute the fraction of policy points at or extremely near the lower
    and upper investment bounds.
    """
    dp = get_dp_settings(config)
    i_min = float(dp["investment_min"])
    i_max = float(dp["investment_max"])
    tol_low = float(dp["policy_lower_bound_tolerance"])
    tol_high = float(dp["policy_upper_bound_tolerance"])

    lower_hits = np.isclose(policy_i, i_min, atol=tol_low)
    upper_hits = np.isclose(policy_i, i_max, atol=tol_high)

    return {
        "share_lower_bound": float(np.mean(lower_hits)),
        "share_upper_bound": float(np.mean(upper_hits)),
        "share_any_bound": float(np.mean(lower_hits | upper_hits)),
        "share_interior": float(np.mean(~(lower_hits | upper_hits))),
    }


def solve_investment_dp(theta, config: dict, fixed_params: dict):
    """
    Solve the minimal dynamic investment model with states (k, z)
    and control k' using value function iteration.

    Parameters
    ----------
    theta : array-like
        theta[0] = psi
        theta[1] = lambda

    config : dict
        Project configuration.

    fixed_params : dict
        Fixed model parameters.

    Returns
    -------
    dict containing:
    - k_grid
    - log_z_grid
    - z_grid
    - z_transition
    - value_function
    - policy_kprime
    - policy_investment
    - policy_profit
    - policy_external_finance
    - policy_adjustment_cost
    - converged
    - n_iter
    - max_diff
    - share_lower_bound
    - share_upper_bound
    - share_any_bound
    - share_interior
    """
    theta = np.asarray(theta, dtype=float)

    dp = get_dp_settings(config)
    beta = float(dp["beta"])
    delta = float(fixed_params["delta"])

    k_grid = build_k_grid(config)
    log_z_grid, z_grid, z_transition = build_z_process(config, fixed_params)

    nk = len(k_grid)
    nz = len(z_grid)

    V = np.zeros((nk, nz), dtype=float)
    policy_kprime = np.zeros((nk, nz), dtype=float)
    policy_i = np.zeros((nk, nz), dtype=float)
    policy_profit = np.zeros((nk, nz), dtype=float)
    policy_ext_fin = np.zeros((nk, nz), dtype=float)
    policy_adj_cost = np.zeros((nk, nz), dtype=float)

    converged = False
    max_diff = np.inf

    for it in range(dp["max_iter"]):
        V_new = np.zeros_like(V)

        for ik, k in enumerate(k_grid):
            kprime_grid = build_kprime_grid(
                config=config,
                current_k=k,
                delta=delta,
            )

            for iz, z in enumerate(z_grid):
                best_value = -np.inf
                best_kprime = kprime_grid[0]
                best_i = investment_rate_from_kprime(
                    k=k,
                    kprime=kprime_grid[0],
                    delta=delta,
                )
                best_profit = 0.0
                best_ext_fin = 0.0
                best_adj_cost = 0.0

                for kprime in kprime_grid:
                    period = one_period_payoff_from_kprime(
                        k=k,
                        z=z,
                        kprime=kprime,
                        theta=theta,
                        fixed_params=fixed_params,
                    )

                    continuation = expected_continuation_value(
                        k_next=kprime,
                        iz=iz,
                        V=V,
                        k_grid=k_grid,
                        z_transition=z_transition,
                    )

                    candidate_value = period["payoff"] + beta * continuation

                    if candidate_value > best_value:
                        best_value = candidate_value
                        best_kprime = kprime
                        best_i = period["investment_rate"]
                        best_profit = period["profit"]
                        best_ext_fin = period["external_finance"]
                        best_adj_cost = period["adjustment_cost"]

                V_new[ik, iz] = best_value
                policy_kprime[ik, iz] = best_kprime
                policy_i[ik, iz] = best_i
                policy_profit[ik, iz] = best_profit
                policy_ext_fin[ik, iz] = best_ext_fin
                policy_adj_cost[ik, iz] = best_adj_cost

        max_diff = float(np.max(np.abs(V_new - V)))
        V = V_new

        if max_diff < dp["tol"]:
            converged = True
            break

    bound_stats = compute_policy_bound_shares(policy_i=policy_i, config=config)

    return {
        "k_grid": k_grid,
        "log_z_grid": log_z_grid,
        "z_grid": z_grid,
        "z_transition": z_transition,
        "value_function": V,
        "policy_kprime": policy_kprime,
        "policy_investment": policy_i,
        "policy_profit": policy_profit,
        "policy_external_finance": policy_ext_fin,
        "policy_adjustment_cost": policy_adj_cost,
        "converged": converged,
        "n_iter": it + 1,
        "max_diff": max_diff,
        "share_lower_bound": bound_stats["share_lower_bound"],
        "share_upper_bound": bound_stats["share_upper_bound"],
        "share_any_bound": bound_stats["share_any_bound"],
        "share_interior": bound_stats["share_interior"],
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
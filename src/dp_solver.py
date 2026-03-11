import copy
import math

import numpy as np

from src.model import (
    profit,
    adjustment_cost,
    net_investment,
    external_finance_needed_with_cash,
    external_finance_cost,
    period_payoff_cash_model,
)


def _apply_debug_overrides(config: dict, block_name: str, base_cfg: dict) -> dict:
    """
    Apply validation-mode-specific debug overrides to a config block.

    Expected structure in settings.json:
        config["debug_overrides"][validation_mode][block_name]

    If debug_mode is False, no overrides are applied.
    """
    out = copy.deepcopy(base_cfg)

    if not bool(config.get("debug_mode", False)):
        return out

    validation_mode = config.get("validation_mode", "fast")
    debug_overrides = config.get("debug_overrides", {})
    mode_overrides = debug_overrides.get(validation_mode, {})

    block_overrides = mode_overrides.get(block_name, {})
    if isinstance(block_overrides, dict):
        out.update(block_overrides)

    return out


def get_dp_settings(config: dict) -> dict:
    """
    Read dynamic-programming settings from config.

    If debug_mode is active, apply the selected validation_mode override from:
        config["debug_overrides"][validation_mode]["dp"]
    """
    dp_cfg = copy.deepcopy(config.get("dp", {}))
    dp_cfg = _apply_debug_overrides(config, "dp", dp_cfg)

    return {
        "beta": float(dp_cfg.get("beta", 0.95)),
        "k_grid_size": int(dp_cfg.get("k_grid_size", 41)),
        "k_min": float(dp_cfg.get("k_min", 10.0)),
        "k_max": float(dp_cfg.get("k_max", 300.0)),
        "p_grid_size": int(dp_cfg.get("p_grid_size", 25)),
        "p_min": float(dp_cfg.get("p_min", 0.0)),
        "p_max": float(dp_cfg.get("p_max", 60.0)),
        "z_grid_size": int(dp_cfg.get("z_grid_size", 7)),
        "tauchen_m": float(dp_cfg.get("tauchen_m", 2.5)),
        "control_grid_size": int(dp_cfg.get("control_grid_size", 181)),
        "cash_control_grid_size": int(dp_cfg.get("cash_control_grid_size", 21)),
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
    """
    if n < 2:
        raise ValueError("Tauchen discretization requires n >= 2.")

    sigma_x = sigma / np.sqrt(1.0 - rho**2)
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

    k_min = float(dp["k_min"])
    k_max = float(dp["k_max"])
    n = int(dp["k_grid_size"])

    grid = np.exp(np.linspace(np.log(k_min), np.log(k_max), n))
    return grid.astype(float)


def _dense_linear_grid(x_min: float, x_max: float, n: int) -> np.ndarray:
    """
    Dense-near-zero linearized grid using a squared spacing transform.
    """
    u = np.linspace(0.0, 1.0, n)
    grid = x_min + (x_max - x_min) * (u**2)
    return grid.astype(float)


def build_p_grid(config: dict) -> np.ndarray:
    """
    Cash grid.

    We use a grid denser near zero because the economically relevant region for
    the cash choice is often concentrated near low cash balances.
    """
    dp = get_dp_settings(config)

    p_min = float(dp["p_min"])
    p_max = float(dp["p_max"])
    n = int(dp["p_grid_size"])

    return _dense_linear_grid(p_min, p_max, n)


def build_kprime_grid(config: dict, current_k: float, delta: float) -> np.ndarray:
    """
    Control grid for next-period capital k'.
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


def build_pprime_grid(config: dict, current_p: float) -> np.ndarray:
    """
    Control grid for next-period cash p'.

    We keep p' within the global support but concentrate resolution near zero
    and around the current cash state.
    """
    dp = get_dp_settings(config)

    p_min = float(dp["p_min"])
    p_max = float(dp["p_max"])
    n = int(dp["cash_control_grid_size"])

    upper = min(max(2.0 * current_p + 10.0, p_min + 1.0), p_max)
    return _dense_linear_grid(p_min, upper, n)


def build_z_process(config: dict, fixed_params: dict):
    """
    Discretize productivity in logs and map to levels.
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


def investment_rate_from_kprime(k: float, kprime: float, delta: float):
    """
    Recover investment rate from chosen next-period capital:

        i = k'/k - (1 - delta)
    """
    k = max(float(k), 1e-8)
    kprime = max(float(kprime), 1e-8)

    return float((kprime / k) - (1.0 - delta))


def one_period_payoff_from_controls(k, p, z, kprime, pprime, theta, fixed_params):
    """
    One-period payoff for the repaired structural cash model.

    State:
        (k, p, z)

    Controls:
        (k', p')

    Payoff is evaluated using the shared model primitives so the solver and
    simulator are guaranteed to use the same timing conventions.
    """
    theta = np.asarray(theta, dtype=float)
    psi = float(theta[0])
    lam = float(theta[1])

    alpha = float(fixed_params["alpha"])
    delta = float(fixed_params["delta"])
    profit_intercept = float(fixed_params["profit_intercept"])
    production_scale = float(fixed_params["production_scale"])
    fixed_cost = float(fixed_params["fixed_cost"])
    cash_return_rate = float(fixed_params["cash_return_rate"])

    i_rate = investment_rate_from_kprime(k=k, kprime=kprime, delta=delta)

    prof = profit(
        k=k,
        z=z,
        alpha=alpha,
        profit_intercept=profit_intercept,
        production_scale=production_scale,
        fixed_cost=fixed_cost,
    )

    investment_expenditure = net_investment(
        k=k,
        kprime=kprime,
        delta=delta,
    )

    adj_cost_value = adjustment_cost(
        k=k,
        investment_rate=i_rate,
        psi=psi,
    )

    ext_finance = external_finance_needed_with_cash(
        profits=prof,
        current_cash=p,
        next_cash=pprime,
        investment=investment_expenditure,
        adj_cost=adj_cost_value,
        cash_return_rate=cash_return_rate,
    )

    ext_finance_cost_value = external_finance_cost(
        ext_finance=ext_finance,
        lam=lam,
    )

    payoff = period_payoff_cash_model(
        profits=prof,
        current_cash=p,
        next_cash=pprime,
        investment=investment_expenditure,
        adj_cost=adj_cost_value,
        lam=lam,
        cash_return_rate=cash_return_rate,
    )

    return {
        "payoff": float(payoff),
        "investment_rate": float(i_rate),
        "investment_expenditure": float(investment_expenditure),
        "profit": float(prof),
        "adjustment_cost": float(adj_cost_value),
        "external_finance": float(ext_finance),
        "external_finance_cost": float(ext_finance_cost_value),
        "cash_next": float(pprime),
    }


def linear_interp_1d(x_grid: np.ndarray, y_values: np.ndarray, x: float):
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


def bilinear_interp_2d(x_grid, y_grid, values, x, y):
    """
    Bilinear interpolation on a rectangular grid with flat extrapolation.

    values must have shape (len(x_grid), len(y_grid)).
    """
    x = float(x)
    y = float(y)

    x = min(max(x, x_grid[0]), x_grid[-1])
    y = min(max(y, y_grid[0]), y_grid[-1])

    ix = np.searchsorted(x_grid, x) - 1
    iy = np.searchsorted(y_grid, y) - 1

    ix = max(0, min(ix, len(x_grid) - 2))
    iy = max(0, min(iy, len(y_grid) - 2))

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]

    q00 = values[ix, iy]
    q01 = values[ix, iy + 1]
    q10 = values[ix + 1, iy]
    q11 = values[ix + 1, iy + 1]

    wx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    wy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    val0 = (1.0 - wx) * q00 + wx * q10
    val1 = (1.0 - wx) * q01 + wx * q11

    return float((1.0 - wy) * val0 + wy * val1)


def _bilinear_interp_2d_vectorized(x_grid, y_grid, values, x_points, y_points):
    """
    Vectorized bilinear interpolation on a rectangular grid.
    """
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)

    x = np.clip(x_points, x_grid[0], x_grid[-1])
    y = np.clip(y_points, y_grid[0], y_grid[-1])

    ix = np.searchsorted(x_grid, x, side="right") - 1
    iy = np.searchsorted(y_grid, y, side="right") - 1

    ix = np.clip(ix, 0, len(x_grid) - 2)
    iy = np.clip(iy, 0, len(y_grid) - 2)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]

    q00 = values[ix, iy]
    q01 = values[ix, iy + 1]
    q10 = values[ix + 1, iy]
    q11 = values[ix + 1, iy + 1]

    denom_x = np.where(np.abs(x1 - x0) < 1e-14, 1.0, x1 - x0)
    denom_y = np.where(np.abs(y1 - y0) < 1e-14, 1.0, y1 - y0)

    wx = (x - x0) / denom_x
    wy = (y - y0) / denom_y

    val0 = (1.0 - wx) * q00 + wx * q10
    val1 = (1.0 - wx) * q01 + wx * q11

    return (1.0 - wy) * val0 + wy * val1


def compute_policy_bound_shares(policy_i: np.ndarray, config: dict):
    """
    Compute fraction of investment policy points at or near the lower/upper bounds.
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


def solve_investment_dp(theta, config: dict, fixed_params: dict, initial_value_function=None):
    """
    Solve the repaired structural cash model with:
    - convex capital adjustment costs
    - costly external finance
    - cash as a structural state/control

    Current state:
        (k, p, z)

    Current controls:
        (k', p')

    Minimal speed improvement:
    --------------------------
    If `initial_value_function` is supplied and has the correct shape, use it as
    the initial guess for value function iteration. This warm start can
    materially reduce iterations across nearby theta evaluations.
    """
    theta = np.asarray(theta, dtype=float)

    dp = get_dp_settings(config)

    beta = float(dp["beta"])
    delta = float(fixed_params["delta"])

    k_grid = build_k_grid(config)
    p_grid = build_p_grid(config)
    log_z_grid, z_grid, z_transition = build_z_process(config, fixed_params)

    nk = len(k_grid)
    np_grid = len(p_grid)
    nz = len(z_grid)

    expected_shape = (nk, np_grid, nz)
    if initial_value_function is not None:
        initial_value_function = np.asarray(initial_value_function, dtype=float)
        if initial_value_function.shape == expected_shape and np.isfinite(initial_value_function).all():
            V = initial_value_function.copy()
        else:
            V = np.zeros(expected_shape, dtype=float)
    else:
        V = np.zeros(expected_shape, dtype=float)

    policy_kprime = np.zeros((nk, np_grid, nz), dtype=float)
    policy_pprime = np.zeros((nk, np_grid, nz), dtype=float)
    policy_i = np.zeros((nk, np_grid, nz), dtype=float)
    policy_profit = np.zeros((nk, np_grid, nz), dtype=float)
    policy_investment_expenditure = np.zeros((nk, np_grid, nz), dtype=float)
    policy_adjustment_cost = np.zeros((nk, np_grid, nz), dtype=float)
    policy_external_finance = np.zeros((nk, np_grid, nz), dtype=float)
    policy_external_finance_cost = np.zeros((nk, np_grid, nz), dtype=float)

    alpha = float(fixed_params["alpha"])
    profit_intercept = float(fixed_params["profit_intercept"])
    production_scale = float(fixed_params["production_scale"])
    fixed_cost = float(fixed_params["fixed_cost"])
    cash_return_rate = float(fixed_params["cash_return_rate"])
    psi = float(theta[0])
    lam = float(theta[1])

    q_cash = 1.0 / (1.0 + cash_return_rate)

    converged = False
    max_diff = np.inf

    for it in range(dp["max_iter"]):
        V_new = np.zeros_like(V)

        # Precompute:
        #   EV(k', p', iz) = E[V(k', p', z') | current z = z_iz]
        expected_value_by_z = np.tensordot(V, z_transition.T, axes=([2], [0]))

        for ik, k in enumerate(k_grid):
            kprime_grid = build_kprime_grid(
                config=config,
                current_k=k,
                delta=delta,
            )

            for ip, p in enumerate(p_grid):
                pprime_grid = build_pprime_grid(config=config, current_p=p)

                kprime_mesh, pprime_mesh = np.meshgrid(
                    kprime_grid,
                    pprime_grid,
                    indexing="ij",
                )
                kprime_vec = kprime_mesh.ravel()
                pprime_vec = pprime_mesh.ravel()

                investment_rate_vec = (kprime_vec / max(k, 1e-8)) - (1.0 - delta)
                investment_expenditure_vec = kprime_vec - (1.0 - delta) * k
                adj_cost_vec = 0.5 * psi * (investment_rate_vec**2) * k

                for iz, z in enumerate(z_grid):
                    prof = float(
                        profit(
                            k=k,
                            z=z,
                            alpha=alpha,
                            profit_intercept=profit_intercept,
                            production_scale=production_scale,
                            fixed_cost=fixed_cost,
                        )
                    )

                    # IMPORTANT CONSISTENCY FIX:
                    # Match the timing in src/model.py exactly:
                    #
                    #   ext_finance = max(
                    #       investment + adj_cost + q_cash * pprime
                    #       - profits - current_cash,
                    #       0
                    #   )
                    #
                    #   payoff = profits + current_cash - investment - adj_cost
                    #            - q_cash * pprime - lambda * ext_finance
                    #
                    # So the solver and simulator now solve the same model.
                    next_cash_cost_vec = q_cash * pprime_vec

                    ext_finance_vec = np.maximum(
                        investment_expenditure_vec
                        + adj_cost_vec
                        + next_cash_cost_vec
                        - prof
                        - p,
                        0.0,
                    )

                    ext_finance_cost_vec = lam * ext_finance_vec

                    payoff_vec = (
                        prof
                        + p
                        - investment_expenditure_vec
                        - adj_cost_vec
                        - next_cash_cost_vec
                        - ext_finance_cost_vec
                    )

                    continuation_vec = _bilinear_interp_2d_vectorized(
                        x_grid=k_grid,
                        y_grid=p_grid,
                        values=expected_value_by_z[:, :, iz],
                        x_points=kprime_vec,
                        y_points=pprime_vec,
                    )

                    candidate_value_vec = payoff_vec + beta * continuation_vec

                    best_flat_idx = int(np.argmax(candidate_value_vec))
                    best_value = float(candidate_value_vec[best_flat_idx])

                    best_kprime = float(kprime_vec[best_flat_idx])
                    best_pprime = float(pprime_vec[best_flat_idx])
                    best_i = float(investment_rate_vec[best_flat_idx])
                    best_profit = prof
                    best_investment_expenditure = float(
                        investment_expenditure_vec[best_flat_idx]
                    )
                    best_adjustment_cost = float(adj_cost_vec[best_flat_idx])
                    best_external_finance = float(ext_finance_vec[best_flat_idx])
                    best_external_finance_cost = float(
                        ext_finance_cost_vec[best_flat_idx]
                    )

                    V_new[ik, ip, iz] = best_value
                    policy_kprime[ik, ip, iz] = best_kprime
                    policy_pprime[ik, ip, iz] = best_pprime
                    policy_i[ik, ip, iz] = best_i
                    policy_profit[ik, ip, iz] = best_profit
                    policy_investment_expenditure[ik, ip, iz] = (
                        best_investment_expenditure
                    )
                    policy_adjustment_cost[ik, ip, iz] = best_adjustment_cost
                    policy_external_finance[ik, ip, iz] = best_external_finance
                    policy_external_finance_cost[ik, ip, iz] = (
                        best_external_finance_cost
                    )

        max_diff = float(np.max(np.abs(V_new - V)))
        V = V_new

        if max_diff < dp["tol"]:
            converged = True
            break

    bound_stats = compute_policy_bound_shares(policy_i=policy_i, config=config)

    return {
        "k_grid": k_grid,
        "p_grid": p_grid,
        "log_z_grid": log_z_grid,
        "z_grid": z_grid,
        "z_transition": z_transition,
        "value_function": V,
        "policy_kprime": policy_kprime,
        "policy_pprime": policy_pprime,
        "policy_investment": policy_i,
        "policy_profit": policy_profit,
        "policy_investment_expenditure": policy_investment_expenditure,
        "policy_adjustment_cost": policy_adjustment_cost,
        "policy_external_finance": policy_external_finance,
        "policy_external_finance_cost": policy_external_finance_cost,
        "converged": converged,
        "n_iter": it + 1,
        "max_diff": max_diff,
        "share_lower_bound": bound_stats["share_lower_bound"],
        "share_upper_bound": bound_stats["share_upper_bound"],
        "share_any_bound": bound_stats["share_any_bound"],
        "share_interior": bound_stats["share_interior"],
    }


def interpolate_policy_investment(k, p, log_z, solution):
    """
    Evaluate solved investment policy at continuous states using:
    - nearest neighbor in log z
    - bilinear interpolation in (k, p)
    """
    k_grid = solution["k_grid"]
    p_grid = solution["p_grid"]
    log_z_grid = solution["log_z_grid"]
    policy_i = solution["policy_investment"]

    iz = int(np.argmin(np.abs(log_z_grid - log_z)))
    policy_slice = policy_i[:, :, iz]

    return bilinear_interp_2d(
        x_grid=k_grid,
        y_grid=p_grid,
        values=policy_slice,
        x=k,
        y=p,
    )


def interpolate_policy_cash_next(k, p, log_z, solution):
    """
    Evaluate solved next-cash policy p'(k,p,z) at continuous states using:
    - nearest neighbor in log z
    - bilinear interpolation in (k, p)
    """
    k_grid = solution["k_grid"]
    p_grid = solution["p_grid"]
    log_z_grid = solution["log_z_grid"]
    policy_pprime = solution["policy_pprime"]

    iz = int(np.argmin(np.abs(log_z_grid - log_z)))
    policy_slice = policy_pprime[:, :, iz]

    return bilinear_interp_2d(
        x_grid=k_grid,
        y_grid=p_grid,
        values=policy_slice,
        x=k,
        y=p,
    )
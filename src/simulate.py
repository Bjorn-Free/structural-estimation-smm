import copy

import numpy as np
import pandas as pd

from src.dp_solver import (
    solve_investment_dp,
    interpolate_policy_investment,
    interpolate_policy_cash_next,
    get_dp_settings,
)
from src.model import (
    unpack_theta,
    get_fixed_params,
    profit,
    adjustment_cost,
    capital_next,
    external_finance_needed_with_cash,
    external_finance_cost,
    evolve_productivity,
    choose_investment_rate,
    target_debt_ratio,
    target_cash_ratio,
)


def _apply_debug_overrides(config: dict, block_name: str, base_cfg: dict) -> dict:
    """
    Apply validation-mode-specific debug overrides to a config block.

    Expected structure in settings.json:
        config["debug_overrides"][validation_mode][block_name]

    Example:
        config["debug_overrides"]["medium"]["simulation"]
        config["debug_overrides"]["medium"]["dp"]

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


def _get_simulation_settings(config: dict) -> dict:
    """
    Read simulation settings from config.

    If debug_mode is active, apply the selected validation_mode override from:
        config["debug_overrides"][validation_mode]["simulation"]
    """
    sim_cfg = copy.deepcopy(config.get("simulation", {}))
    sim_cfg = _apply_debug_overrides(config, "simulation", sim_cfg)

    return {
        "n_firms": int(sim_cfg.get("n_firms", 2000)),
        "t_periods": int(sim_cfg.get("t_periods", 60)),
        "burn_in": int(sim_cfg.get("burn_in", 10)),
        "seed": int(sim_cfg.get("seed", 123)),
        "initial_capital": float(sim_cfg.get("initial_capital", 100.0)),
        "initial_cash_ratio": float(sim_cfg.get("initial_cash_ratio", 0.10)),
        "initial_log_z": float(sim_cfg.get("initial_log_z", 0.0)),
        "start_year": int(sim_cfg.get("start_year", 2000)),
        "sigma_i": float(sim_cfg.get("sigma_i", 0.20)),
        "sigma_lev": float(sim_cfg.get("sigma_lev", 0.08)),
        "sigma_cash": float(sim_cfg.get("sigma_cash", 0.05)),
        "debt_adjust_speed": float(sim_cfg.get("debt_adjust_speed", 0.15)),
        "cash_adjust_speed": float(sim_cfg.get("cash_adjust_speed", 0.10)),
        "investment_adjust_speed": float(
            sim_cfg.get("investment_adjust_speed", 0.65)
        ),
        "asset_multiplier": float(sim_cfg.get("asset_multiplier", 1.5)),
        "use_dynamic_solver": bool(sim_cfg.get("use_dynamic_solver", False)),
    }


def _simulation_shock_cache_key(config: dict) -> tuple:
    """
    Cache key for common-random-number simulation shocks.

    Shocks depend on:
    - effective simulation sizes
    - random seed

    So if validation_mode changes from fast to medium, the key changes too.
    """
    sim = _get_simulation_settings(config)

    return (
        int(sim["n_firms"]),
        int(sim["t_periods"]),
        int(sim["burn_in"]),
        int(sim["seed"]),
    )


def _draw_simulation_shocks(sim: dict) -> dict:
    """
    Pre-draw all simulation shocks once.

    This implements common random numbers (CRN): every time the SMM objective
    is evaluated at a different parameter vector, the model uses the same
    underlying shock realizations.
    """
    n_firms = sim["n_firms"]
    total_periods = sim["t_periods"] + sim["burn_in"]
    seed = sim["seed"]

    rng = np.random.default_rng(seed)

    shocks = {
        "investment": rng.standard_normal(size=(n_firms, total_periods)),
        "leverage": rng.standard_normal(size=(n_firms, total_periods)),
        "cash": rng.standard_normal(size=(n_firms, total_periods)),
        "productivity": rng.standard_normal(size=(n_firms, total_periods)),
    }

    return shocks


def _get_or_create_simulation_shocks(config: dict) -> dict:
    """
    Retrieve fixed simulation shocks from config if they already exist.
    Otherwise create them once and store them in config.

    The cache key depends on the effective simulation settings, so switching
    validation_mode does not accidentally reuse shocks with the wrong shape.
    """
    cache_key = _simulation_shock_cache_key(config)

    if "_simulation_shocks_cache" not in config:
        config["_simulation_shocks_cache"] = {}

    if cache_key not in config["_simulation_shocks_cache"]:
        sim = _get_simulation_settings(config)
        config["_simulation_shocks_cache"][cache_key] = _draw_simulation_shocks(sim)

    return config["_simulation_shocks_cache"][cache_key]


def _dp_cache_key(theta, config: dict, fixed_params: dict):
    """
    Cache key for solved DP objects.

    Include theta, fixed parameters, and the effective DP settings so policy
    reuse remains valid across validation modes and parameter sweeps.
    """
    theta = tuple(np.round(np.asarray(theta, dtype=float), 10))
    dp = get_dp_settings(config)

    key = (
        theta,
        round(float(fixed_params["alpha"]), 10),
        round(float(fixed_params["delta"]), 10),
        round(float(fixed_params["rho"]), 10),
        round(float(fixed_params["sigma"]), 10),
        round(float(fixed_params["profit_intercept"]), 10),
        round(float(fixed_params["production_scale"]), 10),
        round(float(fixed_params["fixed_cost"]), 10),
        round(float(fixed_params["cash_return_rate"]), 10),
        int(dp["k_grid_size"]),
        round(float(dp["k_min"]), 10),
        round(float(dp["k_max"]), 10),
        int(dp["p_grid_size"]),
        round(float(dp["p_min"]), 10),
        round(float(dp["p_max"]), 10),
        int(dp["z_grid_size"]),
        round(float(dp["tauchen_m"]), 10),
        int(dp["control_grid_size"]),
        int(dp["cash_control_grid_size"]),
        round(float(dp["investment_min"]), 10),
        round(float(dp["investment_max"]), 10),
        int(dp["max_iter"]),
        round(float(dp["tol"]), 14),
        round(float(dp["policy_lower_bound_tolerance"]), 14),
        round(float(dp["policy_upper_bound_tolerance"]), 14),
        bool(config.get("debug_mode", False)),
        str(config.get("validation_mode", "fast")),
    )
    return key


def _get_or_solve_dp(theta, config, fixed_params):
    """
    Cache the solved DP model inside config to avoid resolving it repeatedly
    for the exact same parameter vector and fixed-parameter environment.
    """
    cache_key = _dp_cache_key(theta=theta, config=config, fixed_params=fixed_params)

    if "_dp_cache" not in config:
        config["_dp_cache"] = {}

    if cache_key not in config["_dp_cache"]:
        config["_dp_cache"][cache_key] = solve_investment_dp(
            theta=np.asarray(theta, dtype=float),
            config=config,
            fixed_params=fixed_params,
        )

    return config["_dp_cache"][cache_key]


def simulate_firm_panel(theta, config: dict, solution=None) -> pd.DataFrame:
    """
    Simulate a panel of firms from the repaired structural cash model.

    Current stage:
    - investment is structural and comes from the solved DP when enabled
    - cash is structural and comes from the solved DP when enabled
    - capital adjustment costs are active
    - costly external finance is structural
    - leverage remains a reduced-form placeholder for reporting scaffolding only

    Performance note
    ----------------
    If a solved DP object is supplied via `solution`, simulation reuses the
    policy functions directly and does not solve the DP again. This is the
    standard structural-estimation workflow and can save substantial time in SMM.
    """
    theta = np.asarray(theta, dtype=float)
    theta_named = unpack_theta(theta)
    fixed = get_fixed_params(config, theta=theta)
    sim = _get_simulation_settings(config)
    shocks = _get_or_create_simulation_shocks(config)

    psi = theta_named["psi"]
    lam = theta_named["lam"]

    alpha = fixed["alpha"]
    delta = fixed["delta"]
    rho = fixed["rho"]
    sigma = fixed["sigma"]
    profit_intercept = fixed["profit_intercept"]
    production_scale = fixed["production_scale"]
    fixed_cost = fixed["fixed_cost"]
    cash_return_rate = fixed["cash_return_rate"]

    n_firms = sim["n_firms"]
    t_periods = sim["t_periods"]
    burn_in = sim["burn_in"]
    initial_capital = sim["initial_capital"]
    initial_cash_ratio = sim["initial_cash_ratio"]
    initial_log_z = sim["initial_log_z"]
    start_year = sim["start_year"]

    sigma_i = sim["sigma_i"]
    sigma_lev = sim["sigma_lev"]
    sigma_cash = sim["sigma_cash"]
    debt_adjust_speed = sim["debt_adjust_speed"]
    cash_adjust_speed = sim["cash_adjust_speed"]
    investment_adjust_speed = sim["investment_adjust_speed"]
    asset_multiplier = sim["asset_multiplier"]
    use_dynamic_solver = sim["use_dynamic_solver"]

    dp_solution = None
    if use_dynamic_solver:
        if solution is not None:
            dp_solution = solution
        else:
            dp_solution = _get_or_solve_dp(
                theta=theta,
                config=config,
                fixed_params=fixed,
            )

    rows = []

    for firm_index in range(n_firms):
        firm_id = firm_index + 1
        k = initial_capital
        p = initial_cash_ratio * asset_multiplier * k
        log_z = initial_log_z

        z0 = float(np.exp(log_z))
        leverage = float(target_debt_ratio(z=z0, lam=lam))

        if use_dynamic_solver:
            investment_rate = float(
                interpolate_policy_investment(
                    k=k,
                    p=p,
                    log_z=log_z,
                    solution=dp_solution,
                )
            )
            pprime = float(
                interpolate_policy_cash_next(
                    k=k,
                    p=p,
                    log_z=log_z,
                    solution=dp_solution,
                )
            )
        else:
            investment_rate = float(choose_investment_rate(z=z0, psi=psi, lam=lam))
            cash_ratio = float(target_cash_ratio(z=z0, lam=lam))
            pprime = cash_ratio * asset_multiplier * k

        for t in range(t_periods + burn_in):
            z = float(np.exp(log_z))

            investment_shock = shocks["investment"][firm_index, t]
            leverage_shock = shocks["leverage"][firm_index, t]
            cash_shock = shocks["cash"][firm_index, t]
            productivity_shock = shocks["productivity"][firm_index, t]

            if use_dynamic_solver:
                i_target = float(
                    interpolate_policy_investment(
                        k=k,
                        p=p,
                        log_z=log_z,
                        solution=dp_solution,
                    )
                )
                pprime_target = float(
                    interpolate_policy_cash_next(
                        k=k,
                        p=p,
                        log_z=log_z,
                        solution=dp_solution,
                    )
                )

                investment_rate = float(np.clip(i_target, -0.25, 1.00))
                pprime = max(float(pprime_target), 0.0)

            else:
                i_target = float(choose_investment_rate(z=z, psi=psi, lam=lam))
                investment_rate = float(
                    np.clip(
                        (1.0 - investment_adjust_speed) * investment_rate
                        + investment_adjust_speed * i_target
                        + sigma_i * investment_shock,
                        -0.25,
                        1.00,
                    )
                )

                cash_ratio_target = float(target_cash_ratio(z=z, lam=lam))
                cash_ratio_current = p / max(asset_multiplier * k, 1e-8)

                cash_ratio_next = float(
                    np.clip(
                        (1.0 - cash_adjust_speed) * cash_ratio_current
                        + cash_adjust_speed * cash_ratio_target
                        + sigma_cash * cash_shock,
                        0.00,
                        0.80,
                    )
                )
                pprime = cash_ratio_next * asset_multiplier * k

            lev_target = float(target_debt_ratio(z=z, lam=lam))
            leverage = float(
                np.clip(
                    (1.0 - debt_adjust_speed) * leverage
                    + debt_adjust_speed * lev_target
                    + sigma_lev * leverage_shock,
                    0.00,
                    0.95,
                )
            )

            investment_level = investment_rate * k

            adj_cost_value = float(
                adjustment_cost(
                    k=k,
                    investment_rate=investment_rate,
                    psi=psi,
                )
            )

            operating_profit = float(
                profit(
                    k=k,
                    z=z,
                    alpha=alpha,
                    profit_intercept=profit_intercept,
                    production_scale=production_scale,
                    fixed_cost=fixed_cost,
                )
            )

            ext_fin = float(
                external_finance_needed_with_cash(
                    profits=operating_profit,
                    current_cash=p,
                    next_cash=pprime,
                    investment=investment_level,
                    adj_cost=adj_cost_value,
                    cash_return_rate=cash_return_rate,
                )
            )

            ext_fin_cost = float(
                external_finance_cost(
                    ext_finance=ext_fin,
                    lam=lam,
                )
            )

            assets = asset_multiplier * k
            debt = leverage * assets
            cash = p

            leverage_obs = debt / max(assets, 1e-8)
            cash_ratio_obs = cash / max(assets, 1e-8)
            profitability = operating_profit / max(k, 1e-8)

            if t >= burn_in:
                rows.append(
                    {
                        "gvkey": firm_id,
                        "fyear": start_year + (t - burn_in),
                        "capital": k,
                        "assets": assets,
                        "productivity": z,
                        "investment": investment_rate,
                        "leverage": leverage_obs,
                        "profitability": profitability,
                        "cash_ratio": cash_ratio_obs,
                        "debt": debt,
                        "cash": cash,
                        "cash_next": pprime,
                        "operating_profit": operating_profit,
                        "adjustment_cost": adj_cost_value,
                        "external_finance": ext_fin,
                        "external_finance_cost": ext_fin_cost,
                    }
                )

            k_next = capital_next(
                k=k,
                investment_rate=investment_rate,
                delta=delta,
            )

            k = max(float(k_next), 1e-8)
            p = max(float(pprime), 0.0)

            log_z = float(
                evolve_productivity(
                    log_z=log_z,
                    shock=productivity_shock,
                    rho=rho,
                    sigma=sigma,
                )
            )

    df_sim = pd.DataFrame(rows)

    if df_sim.empty:
        raise ValueError("Simulation produced an empty DataFrame.")

    return df_sim


def simulate_moments(theta, config: dict, solution=None):
    """
    Simulate the model and return the simulated moments.
    """
    from src.moments import compute_moments

    df_sim = simulate_firm_panel(theta, config, solution=solution)
    return compute_moments(df_sim, config)


def simulate_panel(theta, config: dict, solution=None) -> pd.DataFrame:
    """
    Public wrapper that returns the simulated panel.

    This is used by diagnostics and plotting code that needs access to the
    full simulated panel, not just the model moments.
    """
    return simulate_firm_panel(theta, config, solution=solution)
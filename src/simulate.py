import numpy as np
import pandas as pd

from src.dp_solver import solve_investment_dp, interpolate_policy_investment
from src.model import (
    unpack_theta,
    get_fixed_params,
    profit,
    adjustment_cost,
    capital_next,
    external_finance_needed,
    external_finance_cost,
    evolve_productivity,
    choose_investment_rate,
    target_debt_ratio,
    target_cash_ratio,
)


def _get_simulation_settings(config: dict) -> dict:
    """
    Read simulation settings from config.
    If settings are missing, use safe defaults.
    """
    sim_cfg = config.get("simulation", {})

    return {
        "n_firms": int(sim_cfg.get("n_firms", 2000)),
        "t_periods": int(sim_cfg.get("t_periods", 60)),
        "burn_in": int(sim_cfg.get("burn_in", 10)),
        "seed": int(sim_cfg.get("seed", 123)),
        "initial_capital": float(sim_cfg.get("initial_capital", 100.0)),
        "initial_log_z": float(sim_cfg.get("initial_log_z", 0.0)),
        "start_year": int(sim_cfg.get("start_year", 2000)),
        "sigma_i": float(sim_cfg.get("sigma_i", 0.20)),
        "sigma_lev": float(sim_cfg.get("sigma_lev", 0.08)),
        "sigma_cash": float(sim_cfg.get("sigma_cash", 0.05)),
        "debt_adjust_speed": float(sim_cfg.get("debt_adjust_speed", 0.15)),
        "cash_adjust_speed": float(sim_cfg.get("cash_adjust_speed", 0.10)),
        "investment_adjust_speed": float(sim_cfg.get("investment_adjust_speed", 0.65)),
        "asset_multiplier": float(sim_cfg.get("asset_multiplier", 1.5)),
        "use_dynamic_solver": bool(sim_cfg.get("use_dynamic_solver", False)),
    }


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
    """
    if "_simulation_shocks" not in config:
        sim = _get_simulation_settings(config)
        config["_simulation_shocks"] = _draw_simulation_shocks(sim)

    return config["_simulation_shocks"]


def _get_or_solve_dp(theta, config, fixed_params):
    """
    Cache the solved minimal DP model inside config to avoid resolving it
    repeatedly for the exact same parameter vector within one run.
    """
    theta = np.asarray(theta, dtype=float)
    cache_key = tuple(np.round(theta, 10))

    if "_dp_cache" not in config:
        config["_dp_cache"] = {}

    if cache_key not in config["_dp_cache"]:
        config["_dp_cache"][cache_key] = solve_investment_dp(
            theta=theta,
            config=config,
            fixed_params=fixed_params,
        )

    return config["_dp_cache"][cache_key]


def simulate_firm_panel(theta, config: dict) -> pd.DataFrame:
    """
    Simulate a panel of firms from the model.

    Transitional design:
    - investment can come either from the solved dynamic policy (preferred for
      the new structural stage) or from the legacy policy rule
    - leverage and cash remain reduced-form placeholders for now
    """
    theta_named = unpack_theta(theta)
    fixed = get_fixed_params(config)
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

    n_firms = sim["n_firms"]
    t_periods = sim["t_periods"]
    burn_in = sim["burn_in"]
    initial_capital = sim["initial_capital"]
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
        dp_solution = _get_or_solve_dp(theta, config, fixed)

    rows = []

    for firm_index in range(n_firms):
        firm_id = firm_index + 1
        k = initial_capital
        log_z = initial_log_z

        z0 = float(np.exp(log_z))
        leverage = float(target_debt_ratio(z=z0, lam=lam))
        cash_ratio = float(target_cash_ratio(z=z0, lam=lam))

        if use_dynamic_solver:
            investment_rate = float(
                interpolate_policy_investment(
                    k=k,
                    log_z=log_z,
                    solution=dp_solution,
                )
            )
        else:
            investment_rate = float(choose_investment_rate(z=z0, psi=psi, lam=lam))

        for t in range(t_periods + burn_in):
            z = float(np.exp(log_z))

            investment_shock = shocks["investment"][firm_index, t]
            leverage_shock = shocks["leverage"][firm_index, t]
            cash_shock = shocks["cash"][firm_index, t]
            productivity_shock = shocks["productivity"][firm_index, t]

            # Investment dynamics:
            # if dynamic solver is active, use the solved policy directly;
            # otherwise use the legacy policy-rule target with smoothing.
            if use_dynamic_solver:
                i_target = float(
                    interpolate_policy_investment(
                        k=k,
                        log_z=log_z,
                        solution=dp_solution,
                    )
                )
                investment_rate = float(np.clip(i_target, -0.25, 1.00))
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

            cash_target = float(target_cash_ratio(z=z, lam=lam))
            cash_ratio = float(
                np.clip(
                    (1.0 - cash_adjust_speed) * cash_ratio
                    + cash_adjust_speed * cash_target
                    + sigma_cash * cash_shock,
                    0.00,
                    0.80,
                )
            )

            investment_level = investment_rate * k

            adj_cost = float(
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

            assets = asset_multiplier * k
            debt = leverage * assets
            cash = cash_ratio * assets

            debt_change = 0.0
            cash_change = 0.0

            ext_fin = float(
                external_finance_needed(
                    profits=operating_profit,
                    investment=investment_level,
                    adj_cost=adj_cost,
                    debt_change=debt_change,
                    cash_change=cash_change,
                )
            )

            ext_fin_cost = float(
                external_finance_cost(
                    ext_finance=ext_fin,
                    lam=lam,
                )
            )

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
                        "operating_profit": operating_profit,
                        "adjustment_cost": adj_cost,
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


def simulate_moments(theta, config: dict):
    """
    Simulate the model and return the simulated moments.
    """
    from src.moments import compute_moments

    df_sim = simulate_firm_panel(theta, config)
    return compute_moments(df_sim, config)


def simulate_panel(theta, config: dict) -> pd.DataFrame:
    """
    Public wrapper that returns the simulated panel.

    This is used by diagnostics and plotting code that needs access to the
    full simulated panel, not just the model moments.
    """
    return simulate_firm_panel(theta, config)
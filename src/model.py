import numpy as np


DEFAULT_ALPHA = 0.70
DEFAULT_DELTA = 0.15
DEFAULT_RHO = 0.70
DEFAULT_SIGMA = 0.35
DEFAULT_PRODUCTION_SCALE = 0.65
DEFAULT_FIXED_COST = 10.0
DEFAULT_PROFIT_INTERCEPT = 0.00
DEFAULT_CASH_RETURN_RATE = 0.02


def unpack_theta(theta):
    """
    Map estimated parameter vector into named parameters.

    Current stage:
    - theta[0] = psi    -> convex capital adjustment cost
    - theta[1] = lam    -> proportional costly external finance
    - theta[2] = rho    -> productivity persistence
    - theta[3] = sigma  -> productivity volatility
    """
    theta = np.asarray(theta, dtype=float)

    psi = float(theta[0])
    lam = float(theta[1])
    rho = float(theta[2])
    sigma = float(theta[3])

    return {"psi": psi, "lam": lam, "rho": rho, "sigma": sigma}


def get_fixed_params(config=None, theta=None):
    """
    Fixed model parameters shared across simulation and the DP solver.

    Current structural stage:
    - Section 3.1 backbone with capital and productivity
    - plus convex capital adjustment costs
    - plus costly external finance
    - plus cash as a structural state/control

    Operating profits are:

        profit(k, z) = production_scale * z * k^alpha
                       + profit_intercept * k
                       - fixed_cost

    Important upgrade:
    ------------------
    rho and sigma can now come either from:
    1. config["model"] as fallback defaults, or
    2. theta if they are being estimated.

    This keeps the rest of the codebase stable while allowing SMM to estimate:
        theta = (psi, lambda, rho, sigma)
    """
    model_cfg = {}
    if config is not None:
        model_cfg = config.get("model", {})

    rho_value = float(model_cfg.get("rho", DEFAULT_RHO))
    sigma_value = float(model_cfg.get("sigma", DEFAULT_SIGMA))

    if theta is not None:
        theta_named = unpack_theta(theta)
        rho_value = float(theta_named["rho"])
        sigma_value = float(theta_named["sigma"])

    return {
        "alpha": float(model_cfg.get("alpha", DEFAULT_ALPHA)),
        "delta": float(model_cfg.get("delta", DEFAULT_DELTA)),
        "rho": rho_value,
        "sigma": sigma_value,
        "production_scale": float(
            model_cfg.get("production_scale", DEFAULT_PRODUCTION_SCALE)
        ),
        "fixed_cost": float(
            model_cfg.get("fixed_cost", DEFAULT_FIXED_COST)
        ),
        "profit_intercept": float(
            model_cfg.get("profit_intercept", DEFAULT_PROFIT_INTERCEPT)
        ),
        "cash_return_rate": float(
            model_cfg.get("cash_return_rate", DEFAULT_CASH_RETURN_RATE)
        ),
    }


def profit_rate(
    k,
    z,
    alpha,
    profit_intercept,
    production_scale=DEFAULT_PRODUCTION_SCALE,
    fixed_cost=DEFAULT_FIXED_COST,
):
    """
    Profitability rate per unit of capital.
    """
    k = np.maximum(np.asarray(k, dtype=float), 1e-8)
    z = np.maximum(np.asarray(z, dtype=float), 1e-8)

    return (
        production_scale * z * (k ** (alpha - 1.0))
        + profit_intercept
        - fixed_cost / k
    )


def profit(
    k,
    z,
    alpha,
    profit_intercept,
    production_scale=DEFAULT_PRODUCTION_SCALE,
    fixed_cost=DEFAULT_FIXED_COST,
):
    """
    Operating profit level.

        profit(k, z) = production_scale * z * k^alpha
                       + profit_intercept * k
                       - fixed_cost
    """
    k = np.maximum(np.asarray(k, dtype=float), 1e-8)
    z = np.maximum(np.asarray(z, dtype=float), 1e-8)

    return (
        production_scale * z * (k ** alpha)
        + profit_intercept * k
        - fixed_cost
    )


def adjustment_cost(k, investment_rate, psi):
    """
    Convex capital adjustment cost:

        AC = 0.5 * psi * i^2 * k
    """
    k = np.maximum(np.asarray(k, dtype=float), 1e-8)
    investment_rate = np.asarray(investment_rate, dtype=float)

    return 0.5 * psi * (investment_rate ** 2) * k


def capital_next(k, investment_rate, delta):
    """
    Capital accumulation:

        k' = (1 - delta + i) * k
    """
    k = np.maximum(np.asarray(k, dtype=float), 1e-8)
    investment_rate = np.asarray(investment_rate, dtype=float)

    return (1.0 - delta + investment_rate) * k


def net_investment(k, kprime, delta):
    """
    Net investment expenditure implied by choosing next-period capital:

        x = k' - (1 - delta) * k
    """
    k = np.maximum(np.asarray(k, dtype=float), 1e-8)
    kprime = np.maximum(np.asarray(kprime, dtype=float), 1e-8)

    return kprime - (1.0 - delta) * k


def current_cash_resources(cash):
    """
    Current cash holdings available as internal funds this period.

    In the repaired Section 3.3-style implementation, current liquid assets
    enter current resources one-for-one.
    """
    cash = np.asarray(cash, dtype=float)
    return np.maximum(cash, 0.0)


def cash_purchase_price(cash_return_rate):
    """
    Price today of one unit of next-period liquid assets.

    We approximate liquid assets as a one-period bond:
        q_cash = 1 / (1 + r_cash)

    so buying p' units today costs q_cash * p'.
    """
    cash_return_rate = float(cash_return_rate)
    return 1.0 / (1.0 + cash_return_rate)


def cash_purchase_cost(next_cash, cash_return_rate=DEFAULT_CASH_RETURN_RATE):
    """
    Current-period purchase cost of next-period liquid assets.
    """
    next_cash = np.maximum(np.asarray(next_cash, dtype=float), 0.0)
    q_cash = cash_purchase_price(cash_return_rate)
    return q_cash * next_cash


def external_finance_needed(
    profits,
    investment,
    adj_cost,
    debt_change=0.0,
    cash_change=0.0,
):
    """
    Generic positive funding gap:

        gap = investment + adjustment_cost + cash_change
              - profits - debt_change
    """
    gap = investment + adj_cost + cash_change - profits - debt_change
    return np.maximum(gap, 0.0)


def external_finance_needed_with_cash(
    profits,
    current_cash,
    next_cash,
    investment,
    adj_cost,
    cash_return_rate=DEFAULT_CASH_RETURN_RATE,
):
    """
    Positive funding gap for the structural cash model.

    Uses:
        investment + adjustment cost + purchase_cost(next_cash)

    Internal funds:
        profits + current cash holdings

    Therefore:
        ext_finance = max(investment + adj_cost + q_cash * next_cash
                          - profits - current_cash, 0)

    This is closer to the Section 3.3 liquid-asset timing than the previous
    zero-return / one-for-one carry formulation.
    """
    current_cash = current_cash_resources(current_cash)
    next_cash_cost = cash_purchase_cost(next_cash, cash_return_rate)

    gap = investment + adj_cost + next_cash_cost - profits - current_cash
    return np.maximum(gap, 0.0)


def external_finance_cost(ext_finance, lam):
    """
    Proportional external finance cost.
    """
    return lam * np.maximum(ext_finance, 0.0)


def payout(
    profits,
    investment,
    adj_cost,
    lam,
    debt_change=0.0,
    cash_change=0.0,
):
    """
    Net payout after real-side uses and external finance costs.

    Reserved generic form from earlier stages.
    """
    ext_finance = external_finance_needed(
        profits=profits,
        investment=investment,
        adj_cost=adj_cost,
        debt_change=debt_change,
        cash_change=cash_change,
    )

    ext_finance_cost_value = external_finance_cost(ext_finance, lam)

    return (
        profits
        - investment
        - adj_cost
        - cash_change
        + debt_change
        - ext_finance_cost_value
    )


def period_payoff_cash_model(
    profits,
    current_cash,
    next_cash,
    investment,
    adj_cost,
    lam,
    cash_return_rate=DEFAULT_CASH_RETURN_RATE,
):
    """
    One-period payoff for the repaired structural cash model:

        payoff = profits
                 + current_cash
                 - investment
                 - adjustment_cost
                 - q_cash * next_cash
                 - lambda * external_finance

    where:
        q_cash = 1 / (1 + r_cash)

    This is a closer implementation of Section 3.3's idea that liquid assets
    are purchased today and pay off next period.
    """
    current_cash_value = current_cash_resources(current_cash)
    next_cash_cost = cash_purchase_cost(next_cash, cash_return_rate)

    ext_finance = external_finance_needed_with_cash(
        profits=profits,
        current_cash=current_cash,
        next_cash=next_cash,
        investment=investment,
        adj_cost=adj_cost,
        cash_return_rate=cash_return_rate,
    )

    ext_finance_cost_value = external_finance_cost(ext_finance, lam)

    return (
        profits
        + current_cash_value
        - investment
        - adj_cost
        - next_cash_cost
        - ext_finance_cost_value
    )


def evolve_productivity(log_z, shock, rho, sigma):
    """
    AR(1) productivity process in logs:

        log z' = rho * log_z + sigma * eps
    """
    return rho * log_z + sigma * shock


def choose_investment_rate(
    z,
    psi,
    lam=0.0,
    base_investment=0.265,
    z_sensitivity=0.32,
    psi_response_scale=0.60,
    psi_level_penalty=0.012,
    lam_penalty=0.10,
):
    """
    Legacy policy-rule investment function.

    This remains available for fallback/debugging comparisons, but the project
    is transitioning toward a solved dynamic policy from dp_solver.py.
    """
    z_centered = np.log(np.maximum(z, 1e-8))

    effective_z_sensitivity = z_sensitivity / (1.0 + psi_response_scale * psi)
    psi_drag = psi_level_penalty * np.log1p(np.maximum(psi, 0.0))

    raw_i = (
        base_investment
        + effective_z_sensitivity * z_centered
        - psi_drag
        - lam_penalty * lam
    )

    return np.clip(raw_i, -0.25, 1.00)


def target_debt_ratio(
    z,
    lam,
    target=0.30,
    z_sensitivity=0.10,
    lam_penalty=0.55,
):
    """
    Reduced-form placeholder leverage rule.

    Debt is still not structural at the current stage and is retained only to
    preserve the broader simulation/reporting scaffolding.
    """
    z_centered = np.log(np.maximum(z, 1e-8))

    raw_lev = (
        target
        + (z_sensitivity / (1.0 + 0.5 * lam)) * z_centered
        - lam_penalty * lam
    )

    return np.clip(raw_lev, 0.0, 0.95)


def target_cash_ratio(
    z,
    lam,
    base_cash=0.16,
    z_sensitivity=-0.03,
    lam_sensitivity=0.45,
):
    """
    Legacy reduced-form placeholder cash rule.

    This is retained only as a fallback/scaffolding object. The current solved
    model stage treats cash structurally.
    """
    z_centered = np.log(np.maximum(z, 1e-8))

    raw_cash = (
        base_cash
        + lam_sensitivity * lam
        + z_sensitivity * z_centered
    )

    return np.clip(raw_cash, 0.0, 0.80)
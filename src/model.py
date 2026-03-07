import numpy as np


DEFAULT_ALPHA = 0.70
DEFAULT_DELTA = 0.15
DEFAULT_RHO = 0.70
DEFAULT_SIGMA = 0.35
DEFAULT_PRODUCTION_SCALE = 0.65
DEFAULT_FIXED_COST = 10.0
DEFAULT_PROFIT_INTERCEPT = 0.00


def unpack_theta(theta):
    """
    Map estimated parameter vector into named parameters.

    Current stage:
    - theta[0] = psi  -> convex capital adjustment cost
    - theta[1] = lam  -> proportional costly external finance
    """
    psi = float(theta[0])
    lam = float(theta[1])
    return {"psi": psi, "lam": lam}


def get_fixed_params(config=None):
    """
    Fixed model parameters shared across simulation and the DP solver.

    Current structural stage:
    - Section 3.1 backbone with capital and productivity
    - plus convex capital adjustment costs
    - plus costly external finance
    - no structural cash yet

    Operating profits are:

        profit(k, z) = production_scale * z * k^alpha
                       + profit_intercept * k
                       - fixed_cost
    """
    model_cfg = {}
    if config is not None:
        model_cfg = config.get("model", {})

    return {
        "alpha": float(model_cfg.get("alpha", DEFAULT_ALPHA)),
        "delta": float(model_cfg.get("delta", DEFAULT_DELTA)),
        "rho": float(model_cfg.get("rho", DEFAULT_RHO)),
        "sigma": float(model_cfg.get("sigma", DEFAULT_SIGMA)),
        "production_scale": float(
            model_cfg.get("production_scale", DEFAULT_PRODUCTION_SCALE)
        ),
        "fixed_cost": float(
            model_cfg.get("fixed_cost", DEFAULT_FIXED_COST)
        ),
        "profit_intercept": float(
            model_cfg.get("profit_intercept", DEFAULT_PROFIT_INTERCEPT)
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


def period_payoff_adjustment_model(profits, investment, adj_cost):
    """
    One-period payoff for the model with capital adjustment costs only:

        payoff = profits - net_investment - adjustment_cost
    """
    profits = np.asarray(profits, dtype=float)
    investment = np.asarray(investment, dtype=float)
    adj_cost = np.asarray(adj_cost, dtype=float)

    return profits - investment - adj_cost


def external_finance_needed(
    profits,
    investment,
    adj_cost,
    debt_change=0.0,
    cash_change=0.0,
):
    """
    Positive funding gap:

        gap = investment + adjustment_cost + cash_change
              - profits - debt_change

        ext_finance = max(gap, 0)
    """
    gap = investment + adj_cost + cash_change - profits - debt_change
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
    Net payout after real-side uses and external finance costs:

        ext_finance = max(investment + adj_cost + cash_change
                          - profits - debt_change, 0)

        payout = profits
                 - investment
                 - adj_cost
                 - cash_change
                 + debt_change
                 - lam * ext_finance

    In the current stage:
    - debt_change = 0
    - cash_change = 0
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


def period_payoff_financing_model(profits, investment, adj_cost, lam):
    """
    One-period payoff for the current model stage:

        payoff = profits - investment - adjustment_cost - external_finance_cost

    where external finance is raised only when internal funds are insufficient.
    """
    return payout(
        profits=profits,
        investment=investment,
        adj_cost=adj_cost,
        lam=lam,
        debt_change=0.0,
        cash_change=0.0,
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

    Debt is not structural at the current stage and is retained only to
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
    Reduced-form placeholder cash rule.

    Cash is not structural at the current stage and is retained only to
    preserve the broader simulation/reporting scaffolding.
    """
    z_centered = np.log(np.maximum(z, 1e-8))

    raw_cash = (
        base_cash
        + lam_sensitivity * lam
        + z_sensitivity * z_centered
    )

    return np.clip(raw_cash, 0.0, 0.80)
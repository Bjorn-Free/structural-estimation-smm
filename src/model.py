import numpy as np


def unpack_theta(theta):
    """
    Map estimated parameter vector into named parameters.

    theta[0] = psi  -> convex adjustment cost
    theta[1] = lam  -> proportional external finance cost
    """
    psi = float(theta[0])
    lam = float(theta[1])
    return {"psi": psi, "lam": lam}


def get_fixed_params(config=None):
    """
    Fixed model parameters.

    This project is transitioning from a policy-rule approximation to a solved
    dynamic model. These are the core economic primitives shared across both.
    """
    return {
        "alpha": 0.70,
        "delta": 0.12,
        "rho": 0.70,
        "sigma": 0.45,
        "profit_intercept": -0.11,
        "profit_shock_scale": 0.25,
    }


def profit_rate(z, alpha, profit_intercept):
    """
    Profitability rate per unit of capital.
    """
    z = np.maximum(z, 1e-8)
    return profit_intercept + (z ** alpha) - 1.0


def profit(k, z, alpha, profit_intercept):
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


def capital_next(k, investment_rate, delta):
    """
    Capital accumulation:

        k' = (1 - delta + i) * k
    """
    return (1.0 - delta + investment_rate) * k


def external_finance_needed(profits, investment, adj_cost, debt_change=0.0, cash_change=0.0):
    """
    Positive funding gap.
    """
    gap = investment + adj_cost + cash_change - profits - debt_change
    return np.maximum(gap, 0.0)


def external_finance_cost(ext_finance, lam):
    """
    Proportional external finance cost.
    """
    return lam * np.maximum(ext_finance, 0.0)


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
    is now transitioning toward a solved dynamic policy from dp_solver.py.
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
    Target leverage ratio.

    This is still a reduced-form placeholder. Debt will become a true choice
    variable once the model expands from the minimal (k,z) dynamic stage to
    the debt-augmented state space.
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
    Target cash ratio.

    This is still a reduced-form placeholder. Cash will become a true state
    and control variable once the model expands to the full structural version.
    """
    z_centered = np.log(np.maximum(z, 1e-8))

    raw_cash = (
        base_cash
        + lam_sensitivity * lam
        + z_sensitivity * z_centered
    )

    return np.clip(raw_cash, 0.0, 0.80)
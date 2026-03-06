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

    Designed to roughly match empirical profitability moments.
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
    base_investment=0.20,
    z_sensitivity=0.55,
    psi_penalty=0.035,
    lam_penalty=0.12,
):
    """
    Investment policy rule.

    Economic interpretation:
    - higher productivity raises investment
    - higher adjustment costs (psi) reduce investment
    - higher external finance costs (lam) also reduce investment

    The productivity response is dampened by psi so that high adjustment
    costs make investment less responsive to shocks.
    """
    z_centered = np.log(np.maximum(z, 1e-8))

    raw_i = (
        base_investment
        + (z_sensitivity / (1.0 + psi)) * z_centered
        - psi_penalty * psi
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

    Economic interpretation:
    - more productive firms can sustain somewhat more debt
    - higher external finance costs reduce target leverage materially

    This makes lambda affect leverage moments more strongly than before.
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

    Economic interpretation:
    - higher external finance costs increase precautionary cash holdings
    - firms with stronger productivity realizations hold slightly less cash

    This gives lambda a stronger precautionary-savings channel.
    """
    z_centered = np.log(np.maximum(z, 1e-8))

    raw_cash = (
        base_cash
        + lam_sensitivity * lam
        + z_sensitivity * z_centered
    )

    return np.clip(raw_cash, 0.0, 0.80)
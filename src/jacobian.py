import numpy as np

from src.moments import moment_covariance_matrix
from src.simulate import simulate_moments


def numerical_jacobian(theta_hat, config, relative_step=1e-2, minimum_step=1e-4):
    """
    Compute the numerical Jacobian of the simulated moment vector with
    respect to the parameter vector using central differences.

    Course alignment
    ----------------
    - Uses a two-sided derivative.
    - Uses parameter-specific step sizes.
    - Relies on common random numbers already embedded in the simulation
      pipeline so the simulated moments vary smoothly across evaluations.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    n_params = theta_hat.size

    base_moments = np.asarray(simulate_moments(theta_hat, config), dtype=float)
    n_moments = base_moments.size

    J = np.zeros((n_moments, n_params), dtype=float)

    for j in range(n_params):
        h = max(minimum_step, relative_step * max(abs(theta_hat[j]), 1.0))

        theta_plus = theta_hat.copy()
        theta_minus = theta_hat.copy()

        theta_plus[j] += h
        theta_minus[j] -= h

        m_plus = np.asarray(simulate_moments(theta_plus, config), dtype=float)
        m_minus = np.asarray(simulate_moments(theta_minus, config), dtype=float)

        if m_plus.shape != base_moments.shape or m_minus.shape != base_moments.shape:
            raise ValueError("Simulated moment vector shape changed during Jacobian evaluation.")

        J[:, j] = (m_plus - m_minus) / (2.0 * h)

    if not np.isfinite(J).all():
        raise ValueError("Jacobian contains non-finite values.")

    return J


def _get_first_available_value(container, candidate_keys, name_for_error):
    """
    Return the first available value from a dictionary given a list of
    candidate keys.
    """
    for key in candidate_keys:
        if key in container:
            return container[key]

    raise KeyError(
        f"Could not find an entry for '{name_for_error}'. "
        f"Tried keys: {candidate_keys}"
    )


def _get_simulation_config(config):
    """
    Return the simulation config block.

    In this project, simulation settings are stored under:
        config["simulation"]
    """
    if "simulation" not in config:
        raise KeyError(
            "Could not find 'simulation' block in config. "
            "Expected simulation settings inside config['simulation']."
        )

    sim_cfg = config["simulation"]

    if not isinstance(sim_cfg, dict):
        raise ValueError("config['simulation'] must be a dictionary.")

    return sim_cfg


def simulation_to_data_ratio(df, config):
    """
    Compute the simulation-to-data size ratio used in the SMM simulation
    noise adjustment factor.

    We interpret:
        j_ratio = N_sim / N_data

    where:
    - N_sim  = number of simulated firm-year observations used for moments
    - N_data = number of empirical firm-year observations used for moments

    With the current project settings:
        N_sim = n_firms * (t_periods - burn_in)
    """
    n_data = len(df)
    if n_data <= 0:
        raise ValueError("Empirical dataset has no observations.")

    sim_cfg = _get_simulation_config(config)

    n_firms = int(
        _get_first_available_value(
            sim_cfg,
            ["n_firms", "num_firms", "number_of_firms"],
            "number of simulated firms",
        )
    )

    n_periods = int(
        _get_first_available_value(
            sim_cfg,
            ["t_periods", "n_periods", "num_periods", "simulation_periods"],
            "number of simulation periods",
        )
    )

    burn_in = int(
        _get_first_available_value(
            sim_cfg,
            ["burn_in", "burnin", "burn_in_periods"],
            "burn-in periods",
        )
    )

    n_sim = n_firms * max(n_periods - burn_in, 1)
    j_ratio = float(n_sim) / float(n_data)

    if j_ratio <= 0 or not np.isfinite(j_ratio):
        raise ValueError("Invalid simulation-to-data ratio.")

    return j_ratio


def smm_parameter_vcov(theta_hat, W, df, config, J=None, use_efficient_formula=True):
    """
    Compute the SMM variance-covariance matrix for the estimated parameters.

    Parameters
    ----------
    theta_hat : array-like
        Estimated parameter vector.

    W : ndarray
        Weighting matrix.

    df : pandas.DataFrame
        Empirical dataset used for data moments.

    config : dict
        Project configuration.

    J : ndarray or None, default None
        Numerical Jacobian. If None, compute it internally.

    use_efficient_formula : bool, default True
        If True, use the efficient-SMM covariance formula with simulation
        adjustment:
            V = (1 + 1 / j_ratio) * (G' W G)^(-1)
        where j_ratio = N_sim / N_data.

        If False, compute the more general sandwich:
            V = (1 + 1 / j_ratio) * (G' W G)^(-1) (G' W S W G) (G' W G)^(-1)
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    W = np.asarray(W, dtype=float)

    if J is None:
        J = numerical_jacobian(theta_hat, config)

    G = np.asarray(J, dtype=float)

    if not np.isfinite(G).all():
        raise ValueError("Jacobian contains non-finite values.")

    bread = G.T @ W @ G

    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        bread_inv = np.linalg.pinv(bread)

    j_ratio = simulation_to_data_ratio(df, config)
    sim_adjustment = 1.0 + (1.0 / j_ratio)

    if use_efficient_formula:
        V = sim_adjustment * bread_inv
    else:
        S = np.asarray(moment_covariance_matrix(df, config), dtype=float)
        meat = G.T @ W @ S @ W @ G
        V = sim_adjustment * (bread_inv @ meat @ bread_inv)

    V = 0.5 * (V + V.T)

    if not np.isfinite(V).all():
        raise ValueError("Parameter variance-covariance matrix contains non-finite values.")

    return V


def smm_standard_errors(theta_hat, W, df, config, J=None, use_efficient_formula=True):
    """
    Compute SMM standard errors, t-statistics, and related diagnostics.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)

    if J is None:
        J = numerical_jacobian(theta_hat, config)

    G = np.asarray(J, dtype=float)
    W = np.asarray(W, dtype=float)

    bread = G.T @ W @ G
    try:
        bread_condition_number = float(np.linalg.cond(bread))
    except np.linalg.LinAlgError:
        bread_condition_number = np.inf

    V = smm_parameter_vcov(
        theta_hat=theta_hat,
        W=W,
        df=df,
        config=config,
        J=G,
        use_efficient_formula=use_efficient_formula,
    )

    diag = np.diag(V).copy()
    diag = np.maximum(diag, 0.0)
    std_errors = np.sqrt(diag)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = theta_hat / std_errors

    j_ratio = simulation_to_data_ratio(df, config)
    sim_adjustment = 1.0 + (1.0 / j_ratio)

    return {
        "jacobian": G,
        "vcov": V,
        "std_errors": std_errors,
        "t_stats": t_stats,
        "simulation_to_data_ratio": j_ratio,
        "simulation_adjustment": sim_adjustment,
        "bread": bread,
        "bread_condition_number": bread_condition_number,
    }
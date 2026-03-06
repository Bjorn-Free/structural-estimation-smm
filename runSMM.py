from src.config import load_config
from src.data import build_compustat, load_clean_data
from src.diagnostics import run_parameter_diagnostics, print_parameter_diagnostics
from src.moments import compute_moments, moment_names
from src.reporting import (
    save_summary_statistics_table,
    save_moment_comparison_table,
    save_parameter_table,
    save_estimation_settings_table,
)
from src.smm import make_weighting_matrix, estimate_smm


def main():
    config = load_config("settings.json")

    if config.get("rebuild_data", False):
        build_compustat(
            config["raw_data_path"],
            config["clean_data_path"],
            config,
        )

    df = load_clean_data(config["clean_data_path"])

    m_data = compute_moments(df, config)
    W = make_weighting_matrix(df, config)

    est = estimate_smm(
        config["theta0"],
        config["bounds"],
        m_data,
        W,
        config,
    )

    labels = moment_names(config)

    summary_df, summary_csv, summary_tex = save_summary_statistics_table(
        df=df,
        decimals=4,
    )

    comparison_df, comparison_csv, comparison_tex = save_moment_comparison_table(
        moment_labels=labels,
        m_data=m_data,
        m_sim=est["m_sim"],
        decimals=4,
    )

    param_df, param_csv, param_tex = save_parameter_table(
        theta_hat=est["theta_hat"],
        param_names=["psi_adjustment_cost", "lambda_external_finance_cost"],
        std_errors=None,
        decimals=4,
    )

    settings_df, settings_csv, settings_tex = save_estimation_settings_table(
        config=config,
        objective_value=est["objective_value"],
        weighting_matrix_name="Identity",
        decimals=4,
    )

    print("Moment names:", labels)
    print("Data moments:", m_data)
    print("Theta hat:", est["theta_hat"])
    print("Optimizer success:", est["success"])
    print("Optimizer message:", est["message"])
    print("Objective value:", est["objective_value"])
    print("Simulated moments:", est["m_sim"])

    # -----------------------------
    # Parameter diagnostics
    # -----------------------------
    theta_hat = est["theta_hat"]

    theta_diagnostics = [
        config["theta0"],
        theta_hat,
        [float(theta_hat[0]) * 1.25, float(theta_hat[1])],   # higher psi
        [float(theta_hat[0]), float(theta_hat[1]) * 1.25],   # higher lambda
    ]

    diagnostic_labels = [
        "initial_guess",
        "theta_hat",
        "high_psi",
        "high_lambda",
    ]

    df_diagnostics = run_parameter_diagnostics(
        config=config,
        theta_list=theta_diagnostics,
        labels=diagnostic_labels,
    )

    print_parameter_diagnostics(df_diagnostics)

    print("\nSummary statistics table:")
    print(summary_df.to_string(index=False))

    print("\nMoment comparison table:")
    print(comparison_df.to_string(index=False))

    print("\nParameter estimates table:")
    print(param_df.to_string(index=False))

    print("\nEstimation settings table:")
    print(settings_df.to_string(index=False))

    print("\nSaved tables:")
    print(f"- {summary_csv}")
    print(f"- {summary_tex}")
    print(f"- {comparison_csv}")
    print(f"- {comparison_tex}")
    print(f"- {param_csv}")
    print(f"- {param_tex}")
    print(f"- {settings_csv}")
    print(f"- {settings_tex}")


if __name__ == "__main__":
    main()
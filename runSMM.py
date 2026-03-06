from src.config import load_config
from src.data import build_compustat, load_clean_data
from src.moments import compute_moments, moment_names
from src.smm import make_weighting_matrix, estimate_smm


def main():
    config = load_config("settings.json")

    # Rebuild cleaned data if requested
    if config.get("rebuild_data", False):
        build_compustat(config["raw_data_path"], config["clean_data_path"], config)

    # Load cleaned data
    df = load_clean_data(config["clean_data_path"])

    # Moments + weighting matrix
    m_data = compute_moments(df, config)
    W = make_weighting_matrix(df, config)

    # SMM estimation (placeholder for now)
    est = estimate_smm(config["theta0"], config["bounds"], m_data, W, config)

    print("Moment names:", moment_names(config))
    print("Data moments:", m_data)
    print("Theta hat:", est["theta_hat"])


if __name__ == "__main__":
    main()
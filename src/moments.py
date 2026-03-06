import numpy as np
import pandas as pd


def moment_names(config: dict) -> list:
    """
    Names of moments used in estimation.
    Placeholder for now.
    """
    return ["mean_assets", "std_assets"]


def compute_moments(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Compute simple placeholder moments from the dataset.
    Later we will replace these with real finance moments.
    """

    if "at" not in df.columns:
        raise ValueError("Column 'at' (assets) not found in dataset")

    mean_assets = df["at"].mean()
    std_assets = df["at"].std()

    return np.array([mean_assets, std_assets])
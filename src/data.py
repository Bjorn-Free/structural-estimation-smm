from pathlib import Path
import pandas as pd


def build_compustat(raw_path: str, clean_path: str, config: dict) -> pd.DataFrame:
    """
    Temporary function: copies raw dataset to cleaned dataset.
    Later we will add real Compustat cleaning here.
    """

    raw_p = Path(raw_path)
    clean_p = Path(clean_path)

    # create clean directory if needed
    clean_p.parent.mkdir(parents=True, exist_ok=True)

    if not raw_p.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_p}")

    df = pd.read_csv(raw_p)

    # for now just copy raw -> clean
    df.to_csv(clean_p, index=False)

    return df


def load_clean_data(clean_path: str) -> pd.DataFrame:
    """
    Load cleaned dataset.
    """

    p = Path(clean_path)

    if not p.exists():
        raise FileNotFoundError(f"Clean data not found at: {p}")

    return pd.read_csv(p)
import pandas as pd


def load_raw_data(data_path: str = "./data/raw.csv") -> pd.DataFrame:
    return pd.read_csv(data_path)

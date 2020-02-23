import numpy as np
import pandas as pd

raw_data_path = "~/Downloads/raw.csv"
output_data_path = "../data"


def main():
    (
        pd.read_csv(raw_data_path)
        .pipe(clean_column_names)
        .pipe(anonymize)
        .pipe(drop_very_low_forces)
        .pipe(replace_nans_by_other_side)
        .to_csv(f"{output_data_path}/raw.csv", index=False)
    )


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(columns=lambda x: x.replace("/G", "/L").replace("/D", "/R"))


def anonymize(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop("Name", axis=1)


def drop_very_low_forces(data: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    to_drop = (data.filter(like="/") < threshold).any(axis=1)
    print(
        f"\tRemove {to_drop.sum()} rows that have less than {threshold} kg test:\n{data.loc[to_drop]}"
    )
    return data.loc[~to_drop]


def replace_nans_by_other_side(data: pd.DataFrame) -> pd.DataFrame:
    nans = data.isna()
    if not nans.any().any():
        return data
    for row, col in np.argwhere(nans.values):
        col_label = data.columns[col]
        col_replacer = col + 1 if col_label.endswith("G") else col - 1
        print(
            f"\t`{col_label} = {data.iloc[row, col]}` replaced by `{data.columns[col_replacer]} = {data.iloc[row, col_replacer]}`"
        )
        data.iloc[row, col] = data.iloc[row, col_replacer]
    return data


if __name__ == "__main__":
    main()

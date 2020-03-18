import pandas as pd


def process_force_data(data: pd.DataFrame, options: dict) -> pd.DataFrame:
    forces = data.filter(like="/").pipe(
        normalize_force_data, data, options["normalization"]
    )

    left = forces.filter(like="/L").rename(columns=lambda x: x[:-2])
    right = forces.filter(like="/R").rename(columns=lambda x: x[:-2])

    processed = aggregate_force_data(left, right, options["aggregation"])

    if options["imbalance"]:
        imbalance = compute_force_imbalance(left, right)
        processed[imbalance.columns] = imbalance

    return data.drop(forces.columns, axis=1).join(processed)


def normalize_force_data(
    forces: pd.DataFrame, data: pd.DataFrame, strategy: str
) -> pd.DataFrame:
    if strategy == "None":
        return forces
    elif strategy == "Weight":
        normalizer = data["Weight"]
    elif strategy == "Weight x Height":
        normalizer = data["Weight"] * data["Height"]
    elif strategy == "IMC":
        normalizer = data["Weight"] / data["Height"] ** 2
    else:
        raise ValueError(f"{strategy} is not a force normalization strategy")
    return forces.divide(normalizer, axis=0)


def aggregate_force_data(
    left: pd.DataFrame, right: pd.DataFrame, strategy: str
) -> pd.DataFrame:
    if strategy == "F-score":
        return (2 * (left * right) / (left + right)).add_prefix("F-score ")
    elif strategy == "Mean":
        return ((left + right) / 2).add_prefix("Mean ")
    else:
        raise ValueError(f"{strategy} is not an aggregation strategy")


def compute_force_imbalance(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return (((left - right) / left).abs() * 100).add_prefix("Imb ")

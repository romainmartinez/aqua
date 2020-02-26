from typing import Tuple

import pandas as pd
from sklearn import model_selection

from aqua._constant import available_targets, random_seed, models_functions


def variables_targets_split(
    data: pd.DataFrame, targets: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return data[targets], data.drop(available_targets, axis=1)


def train_test_split(
    variables: pd.DataFrame, targets: pd.DataFrame, test_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return model_selection.train_test_split(
        variables, targets, test_size=test_size / 100, random_state=random_seed
    )


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, model_name: str):
    return {
        target: models_functions[model_name]().fit(X_train, y_train[target])
        for target in y_train
    }


def predict(X: pd.DataFrame, y: pd.DataFrame, model: dict,) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "target": target,
                    "real": y[target],
                    "predicted": model[target].predict(X),
                }
            )
            for target in y
        ]
    )


def evaluation(predictions: pd.DataFrame) -> pd.DataFrame:
    mae = "MAE = abs(real - predicted)"
    mape = "MAPE = abs((real - predicted) / real) * 100"
    return predictions.eval(mae).eval(mape)

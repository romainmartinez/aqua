from typing import Tuple

import pandas as pd
from sklearn import model_selection

from aqua._constant import available_targets, random_seed


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

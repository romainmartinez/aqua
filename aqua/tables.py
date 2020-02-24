import pandas as pd
import streamlit as st

from aqua._constant import variables_description, targets_description


def mean_and_std_column(x):
    mu = x.mean().round(2).astype(str)
    sigma = x.std().round(2).astype(str)
    return mu + " (+/- " + sigma + ")"


def describe_table(
    data: pd.DataFrame, groupby: list = None, is_targets: bool = False
) -> None:
    if groupby is None:
        groupby = ["variable"]
    describe = data.groupby(groupby).apply(mean_and_std_column)

    if describe.index.nlevels > 1:
        describe = describe.unstack()
        describe.columns = describe.columns.get_level_values(1)
    description = targets_description if is_targets else variables_description
    describe.insert(
        0, "Description", describe.index.map(description),
    )
    st.table(describe)

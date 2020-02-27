import pandas as pd
import streamlit as st

from aqua._constant import variables_description, targets_description


def mean_and_std_column(x):
    mu = x.mean().round(2).astype(str)
    sigma = x.std().round(2).astype(str)
    return mu + " (+/- " + sigma + ")"


def describe_table(
    data: pd.DataFrame, groupby: list = None, description: str = None
) -> None:
    if groupby is None:
        groupby = ["variable"]
    describe = data.groupby(groupby).apply(mean_and_std_column)

    if describe.index.nlevels > 1:
        describe = describe.unstack()
        describe.columns = describe.columns.get_level_values(1)

    if description:
        describe.insert(
            0,
            "Description",
            describe.index.map(
                targets_description
                if description == "targets"
                else variables_description
            ),
        )
    st.dataframe(describe)

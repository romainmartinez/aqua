from typing import Tuple

import streamlit as st

from aqua._constant import available_targets, default_targets, normalization_strategies, available_models, default_models


def make_title() -> None:
    title = "# Predicting eggbeater kick performances from hip joint function testing in artistic swimming"
    authors = "> [__Martinez Romain__](https://github.com/romainmartinez), Monga-Dubreuil Élodie, Assila Najoua, Desmyttere Gauthier and Begon Mickael"
    affiliation = "School of Kinesiology and Exercise Science, Faculty of Medicine, University of Montreal"
    link = "[`Source code and data`](https://github.com/romainmartinez/aqua)"
    # TODO: description
    st.markdown("\n\n".join((title, authors, affiliation, link)))


def make_sidebar() -> Tuple[dict, dict, dict]:
    st.sidebar.markdown("### Data processing options")
    processing_options = {
        "normalization": st.sidebar.selectbox(
            "Force normalization strategy", normalization_strategies
        ),
        "imbalance": st.sidebar.checkbox("Compute left - right imbalance", value=True),
        "aggregation": st.sidebar.selectbox(
            "Left - right aggregation", ["F-score", "Mean"]
        ),
    }
    st.sidebar.markdown(r">$\text{F-score} = 2 \times \frac{L \times R}{L + R}$")

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Modelling options")
    modelling_options = {
        "targets": st.sidebar.multiselect(
            "Target metrics", available_targets, default_targets
        ),
        "test_size": st.sidebar.number_input(
            "Test split size (%)", min_value=0, max_value=100, value=20
        ),
        "models": st.sidebar.multiselect("Models", available_models, default_models)
    }

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Plots options")
    plots_options = {
        "distribution": st.sidebar.selectbox("Distribution", ["KDE", "ECDF"])
    }
    return processing_options, modelling_options, plots_options

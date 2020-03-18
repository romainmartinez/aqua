import altair as alt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from aqua import tables
from aqua._constant import forces_order

empty_axis = alt.Axis(labels=False, ticks=False, domain=False, grid=False)
xaxis = alt.Axis(labelFlush=False)

colors = {"primary": "#f63366", "grey": "#4C566A", "dark": "#2E3440"}
width = 280
height = 75


def plot_kde(
    value: str = "value", variable: str = "variable", **chart_kwargs
) -> alt.Chart:
    dist = (
        alt.Chart(height=height, width=width, **chart_kwargs)
        .transform_density(value, as_=[value, "density"], groupby=[variable])
        .mark_area(color=colors["grey"], opacity=0.6)
        .encode(
            alt.X(value, axis=xaxis), alt.Y("density:Q", title=None, axis=empty_axis)
        )
    )
    point = (
        alt.Chart()
        .mark_circle(size=120, color=colors["dark"], y="height")
        .encode(alt.X(f"mean({value})", title=None, scale=alt.Scale(zero=False)))
    )
    bar = (
        alt.Chart()
        .mark_rule(size=5, color=colors["dark"], y="height")
        .encode(alt.X(f"q1({value})"), alt.X2(f"q3({value})"))
    )
    return dist + bar + point


def plot_anthropometry(variables: pd.DataFrame) -> None:
    anthropo = variables[["Height", "Weight"]].melt()

    tables.describe_table(anthropo, description="variables")

    plots = (
        plot_kde()
        .facet(data=anthropo, column=alt.Column("variable", title=None))
        .resolve_scale(x="independent", y="independent")
    )
    st.altair_chart(plots, use_container_width=True)
    # TODO caption?


def plot_forces(variables: pd.DataFrame) -> None:
    forces = variables.drop(["Height", "Weight"], axis=1).melt()
    forces[["type", "variable"]] = forces["variable"].str.split(expand=True)

    tables.describe_table(forces, groupby=["variable", "type"], description="variables")
    row_kwargs = dict(shorthand="variable", title=None, sort=forces_order)
    column = alt.Column("type", title=None)

    forces_plot = (
        plot_kde()
        .facet(
            data=forces.query("type != 'Imb'"),
            row=alt.Row(
                header=alt.Header(labelAngle=0, labelAlign="left"), **row_kwargs
            ),
            column=column,
        )
        .resolve_scale(y="independent")
        .properties(bounds="flush")
    )

    imb_plot = (
        plot_kde()
        .facet(
            data=forces.query("type == 'Imb'"),
            row=alt.Row(header=alt.Header(labelFontSize=0), **row_kwargs),
            column=column,
        )
        .resolve_scale(y="independent")
        .properties(bounds="flush")
    )

    plots = (forces_plot | imb_plot).configure_facet(spacing=5)
    st.altair_chart(plots)


def plot_targets(targets: pd.DataFrame) -> None:
    targets_melted = targets.melt()

    tables.describe_table(targets_melted, description="targets")

    dist_plot = (
        plot_kde()
        .facet(
            data=targets_melted,
            row=alt.Row(
                "variable",
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
        )
        .configure_facet(spacing=5)
        .resolve_scale(y="independent")
        .properties(bounds="flush")
    )
    st.altair_chart(dist_plot)


def plot_error_dist(predictions: pd.DataFrame) -> None:
    predictions_melted = predictions.melt(id_vars="target", value_vars=["MAE", "MAPE"])
    tables.describe_table(predictions_melted, groupby=["target", "variable"])

    row_kwargs = dict(shorthand="target", title=None, sort=forces_order)
    column = alt.Column("variable", title=None)

    mae = (
        plot_kde()
        .facet(
            data=predictions_melted.query("variable == 'MAE'"),
            row=alt.Row(
                header=alt.Header(labelAngle=0, labelAlign="left"), **row_kwargs
            ),
            column=column,
        )
        .resolve_scale(y="independent")
        .properties(bounds="flush")
    )

    mape = (
        plot_kde()
        .facet(
            data=predictions_melted.query("variable == 'MAPE'"),
            row=alt.Row(header=alt.Header(labelFontSize=0), **row_kwargs),
            column=column,
        )
        .resolve_scale(y="independent")
        .properties(bounds="flush")
    )

    plots = (mae | mape).configure_facet(spacing=5)
    st.altair_chart(plots)


def plot_correlation_matrix(variables: pd.DataFrame, targets: pd.DataFrame) -> None:
    data = variables.join(targets).corr()
    col_order = data.columns.to_list()

    half_corr = (
        data.where(np.triu(np.ones(data.shape)).astype(np.bool))
        .reset_index()
        .melt(id_vars="index")
        .dropna()
    )

    plot_dimension = 600
    corr = (
        alt.Chart(half_corr, width=plot_dimension, height=plot_dimension)
        .mark_rect()
        .encode(
            alt.X("index", sort=col_order, title=None),
            alt.Y("variable", sort=col_order, title=None),
            alt.Tooltip("value"),
            alt.Color(
                "value", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title=None
            ),
        )
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(corr)


def plot_error_residuals(predictions: pd.DataFrame) -> None:
    points = (
        alt.Chart(predictions.eval("Residuals = predicted - real"))
        .mark_circle(size=100)
        .encode(
            alt.X("predicted", title="Predicted", scale=alt.Scale(zero=False)),
            alt.Y("Residuals", title="Residuals"),
            alt.Color("target"),
        )
    )

    rule = alt.Chart(pd.DataFrame([{"zero": 0}])).mark_rule().encode(alt.Y("zero"))

    st.altair_chart(points + rule, use_container_width=True)


def model_comparison(predictions_list: list):
    predictions_melted = pd.concat(predictions_list).melt(
        id_vars="model", value_vars=["MAE", "MAPE"]
    )

    tables.describe_table(predictions_melted, groupby=["model", "variable"])

    row_kwargs = dict(shorthand="model", title=None, sort=forces_order)
    column = alt.Column("variable", title=None)

    mae = (
        plot_kde()
        .facet(
            data=predictions_melted.query("variable == 'MAE'"),
            row=alt.Row(
                header=alt.Header(labelAngle=0, labelAlign="left"), **row_kwargs
            ),
            column=column,
        )
        .resolve_scale(y="independent")
        .properties(bounds="flush")
    )

    mape = (
        plot_kde()
        .facet(
            data=predictions_melted.query("variable == 'MAPE'"),
            row=alt.Row(header=alt.Header(labelFontSize=0), **row_kwargs),
            column=column,
        )
        .resolve_scale(y="independent")
        .properties(bounds="flush")
    )

    plots = (mae | mape).configure_facet(spacing=5)
    st.altair_chart(plots)


def plot_shap_values(X: pd.DataFrame, model: dict) -> pd.DataFrame:
    target = "EB mean force"
    # st.pyplot(
    #     shap.summary_plot(shap.TreeExplainer(model[target], data=X).shap_values(X), X)
    # )

    shap_values = pd.DataFrame(
        shap.TreeExplainer(model[target], data=X).shap_values(X), columns=X.columns
    )

    y_order = shap_values.abs().mean().nlargest(6).index.to_list()
    shap_values = shap_values[y_order].melt()
    # shap_values["rank"] = X.rank().melt()["value"].values
    shap_values["Z-score"] = ((X[y_order] - X[y_order].mean()) / X[y_order].std()).melt()["value"].clip(-0.5, 0.5)

    # dist = (
    #     alt.Chart(shap_values)
    #     .mark_circle(size=100)
    #     .encode(
    #         alt.X("value", title=None),
    #         alt.Y("variable", title=None, sort=y_order),
    #         alt.Color("Z-score", scale=alt.Scale(scheme="redblue", domain=[-2.5, 2.5])),
    #     )
    # )
    # rule = alt.Chart(pd.DataFrame([{'zero': 0}])).mark_rule().encode(alt.X('zero'))


    stripplot = alt.Chart(shap_values, height=20, width=width).mark_circle(size=100, clip=True).encode(
        alt.Y(
            'jitter:Q',
            title=None,
            axis=alt.Axis(values=[0], ticks=False, grid=False, labels=False),
        ),
        alt.X('value', title="Shap value", scale=alt.Scale(domain=[-.4, .4])),
        alt.Color("Z-score", scale=alt.Scale(scheme="redblue", domain=[-0.5, 0.5])),
        alt.Row(
            'variable',
            title=None,
            sort=y_order,
            header=alt.Header(
                labelAngle=0,
                labelAlign='left',
            ),
        ),
    ).transform_calculate(
        jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )



    st.altair_chart(stripplot)

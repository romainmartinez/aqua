import altair as alt
import pandas as pd
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
    height = 75

    forces_plot = (
        plot_kde()
        .facet(
            data=forces.query("type != 'Imb.'"),
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
            data=forces.query("type == 'Imb.'"),
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
    height = 75

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

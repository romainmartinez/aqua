import streamlit as st

from aqua import ui, data, processing, ml, plots

ui.make_title()
processing_options, modelling_options = ui.make_sidebar()

raw_data = data.load_raw_data().pipe(processing.process_force_data, processing_options)

st.markdown("## 1. Data description")

if st.checkbox("Show raw data", key="show-raw"):
    st.dataframe(raw_data)

targets, variables = ml.variables_targets_split(raw_data, modelling_options["targets"])

st.markdown("### 1.1 Variables")
plots.plot_anthropometry(variables)
plots.plot_forces(variables)

st.markdown("### 1.2 Targets")
plots.plot_targets(targets)

st.markdown("### 1.3 Correlation matrix")
plots.plot_correlation_matrix(variables, targets)

st.markdown("## 2. Data modelling")
X_train, X_test, y_train, y_test = ml.train_test_split(
    variables, targets, modelling_options["test_size"]
)

st.markdown(
    f"Train split size: `{X_train.shape[0]}` ({X_train.shape[0] / raw_data.shape[0]:.2f}%)"
)
st.markdown(
    f"Test split size: `{X_test.shape[0]}` ({X_test.shape[0] / raw_data.shape[0]:.2f}%)"
)

st.markdown("### 2.1 Test split evaluation")

# Dirty from here
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import numpy as np

rename_cols = lambda x: x.lower().replace(" ", "_").replace("-", "_")
train = pd.concat(
    [(X_train - X_train.mean()) / X_train.std(), y_train], axis="columns"
).rename(columns=rename_cols)
var_cols = X_train.rename(columns=rename_cols).columns
target = 'eb_mean_height'

with pm.Model() as model:
    pm.GLM.from_formula(f"{target} ~ {' + '.join(var_cols)}", train)
    trace = pm.sample(5000)

az.plot_trace(trace)
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
ax.axvline(x=0, color="black", linewidth=3)
az.plot_forest(
    trace,
    kind="ridgeplot",
    combined=True,
    var_names=var_cols,
    colors="#800000",
    ridgeplot_overlap=1.2,
    ridgeplot_alpha=0.6,
    linewidth=0,
    show=False,
    ax=ax,
)
az.plot_forest(
    trace,
    kind="forestplot",
    colors="black",
    combined=True,
    var_names=var_cols,
    ridgeplot_overlap=1.2,
    show=False,
    ax=ax,
    linewidth=3,
    credible_interval=0.9,
)
ax.set_xlim((-1, 1))
ax.set_title("")
plt.show()

ppc = pm.sample_posterior_predictive(trace, samples=10, model=model)
data_ppc = az.from_pymc3(trace=trace, posterior_predictive=ppc)

az.plot_ppc(data_ppc, alpha=0.5)
plt.show()

az.plot_kde(train[target])
az.plot_kde(ppc['y'])

for sample in range(ppc['y'].shape[0]):
    y_vals, lower, upper = az._fast_kde(ppc["y"][sample, :])
    x_vals = np.linspace(lower, upper, y_vals.shape[0])
    plt.plot(x_vals, y_vals, color='black')
plt.show()


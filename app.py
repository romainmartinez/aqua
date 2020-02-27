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

predictions_list = []
for model_name in modelling_options["models"]:
    st.markdown(f"#### {model_name}")

    model = ml.train_model(X_train, y_train, model_name)
    predictions = ml.predict(X_test, y_test, model).pipe(ml.evaluation)
    plots.plot_error_dist(predictions)
    plots.plot_error_residuals(predictions)

    if "Regression" not in model_name:
        # shap_values = ml.get_shap_values(variables, model, is_tree="Regression" not in model_name)
        plots.plot_shap_values(variables, model)

    predictions_list.append(predictions.assign(model=model_name))

st.markdown("### 2.2 Model comparison")
plots.model_comparison(predictions_list)

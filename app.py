import streamlit as st

from aqua import ui, data, processing

ui.make_title()
processing_options, prediction_options = ui.make_sidebar()

raw_data = data.load_raw_data().pipe(processing.process_force_data, processing_options)


st.markdown("## 1. Data description")

if st.checkbox("Show raw data", key="show-raw"):
    st.dataframe(raw_data)

st.markdown("### 1.1 Targets")


st.markdown("### 1.2 Variables")

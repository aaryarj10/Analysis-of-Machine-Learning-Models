import streamlit as st
import pandas as pd
import sys
import os

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import run_pipeline

st.title("🌱 Green AI Model Comparison")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("Run Analysis"):
        # Save uploaded file
        os.makedirs("data/raw", exist_ok=True)
        file_path = "data/raw/user_dataset.csv"
        df.to_csv(file_path, index=False)

        results_df = run_pipeline(file_path, target_column)

        st.subheader("Results")
        st.dataframe(results_df)

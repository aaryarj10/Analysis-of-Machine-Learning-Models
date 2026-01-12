import streamlit as st
import pandas as pd
import sys
import os

# Page Config
st.set_page_config(page_title="Green AI Dashboard", layout="centered")

# Path to access backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import run_pipeline
from ui.ui_helpers import (
    plot_target_distribution,
    plot_missing_values,
    plot_correlation,
    plot_accuracy,
    plot_energy,
    plot_tradeoff,
)

# AI Model Ranking
def add_model_ranking(results_df):
    metric = "R2 Score" if "R2 Score" in results_df.columns else "Accuracy"

    acc_norm = (results_df[metric] - results_df[metric].min()) / (
        results_df[metric].max() - results_df[metric].min()
    )

    time_norm = (results_df["Training Time (s)"] - results_df["Training Time (s)"].min()) / (
        results_df["Training Time (s)"].max() - results_df["Training Time (s)"].min()
    )

    results_df["Green AI Score"] = (0.7 * acc_norm) + (0.3 * (1 - time_norm))

    ranked_df = results_df.sort_values("Green AI Score", ascending=False)

    st.subheader("🥇 Green AI Model Ranking")
    st.dataframe(ranked_df[["Model", "Green AI Score"]])

    return ranked_df

# Title
st.title("🌱 Green AI Model Comparison")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

# Dataset Preview 
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
    st.caption(f"📌 Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

# Dataset Visualization for feature and target analysis
    with st.expander("📊 Dataset Visualizations"):
        plot_target_distribution(df, target_column)
        plot_missing_values(df)
        plot_correlation(df)

# Run Analysis
    if st.button("🚀 Run Model Analysis"):
        os.makedirs("data/raw", exist_ok=True)
        file_path = "data/raw/user_dataset.csv"
        df.to_csv(file_path, index=False)

        results_df = run_pipeline(file_path, target_column)

# Comparison Table
        st.subheader("📈 Model Comparison Results")
        st.dataframe(results_df)

# Recommendation Cards 
        metric = "R2 Score" if "R2 Score" in results_df.columns else "Accuracy"

        best_accuracy = results_df.loc[results_df[metric].idxmax()]
        best_energy = results_df.loc[results_df["Training Time (s)"].idxmin()]

        st.subheader("🏆 Model Recommendation")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="🎯 Best Accuracy Model",
                value=best_accuracy["Model"],
                delta=f"{metric}: {round(best_accuracy[metric], 3)}"
            )

        with col2:
            st.metric(
                label="⚡ Most Energy Efficient Model",
                value=best_energy["Model"],
                delta=f"Time: {round(best_energy['Training Time (s)'], 3)} s"
            )

# Green AI Ranking Results
        ranked_df = add_model_ranking(results_df)

# Model Performance Visualizations 
        with st.expander("📉 Model Performance Visualizations"):
            plot_accuracy(results_df)
            plot_energy(results_df)
            plot_tradeoff(results_df)

# Download Report
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Model Comparison Report",
            data=csv,
            file_name="green_ai_model_report.csv",
            mime="text/csv"
        )

        st.success("✅ Analysis completed. You can download the report above.")

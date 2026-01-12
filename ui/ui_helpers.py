import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os


# Function to save and show plots
def save_and_show(fig, filename):
    os.makedirs("results/graphs", exist_ok=True)
    fig.savefig(f"results/graphs/{filename}", bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)


# Target Distribution Graph
def plot_target_distribution(df, target):
    st.subheader("📊 Target Variable Distribution")

    fig, ax = plt.subplots(figsize=(5,4))

    if df[target].nunique() <= 10:
        df[target].value_counts().plot(kind="bar", ax=ax, color="#4CAF50")
        ax.set_ylabel("Count")
    else:
        ax.hist(df[target], bins=30, color="#2196F3")
        ax.set_ylabel("Frequency")

    ax.set_xlabel(target)
    ax.set_title("Target Distribution")

    save_and_show(fig, "target_distribution.png")


# Missing Values Graph
def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        st.info("No missing values found.")
        return

    st.subheader("🧹 Missing Values Overview")

    fig, ax = plt.subplots(figsize=(5,4))
    missing.plot(kind="bar", ax=ax, color="#FF9800")
    ax.set_ylabel("Count")
    ax.set_title("Missing Values per Column")

    save_and_show(fig, "missing_values.png")


# Correlation Heatmap Graph
def plot_correlation(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return

    st.subheader("🔗 Feature Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")

    save_and_show(fig, "correlation_heatmap.png")


# Accuracy Comparison Graph
def plot_accuracy(results_df):
    metric = "R2 Score" if "R2 Score" in results_df.columns else "Accuracy"

    st.subheader("🎯 Model Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(results_df["Model"], results_df[metric], color="#673AB7")
    ax.set_ylabel(metric)
    ax.set_title("Accuracy Comparison")
    plt.xticks(rotation=20)

    save_and_show(fig, "accuracy_comparison.png")


# Energy Consumption Comparison Graph
def plot_energy(results_df):
    st.subheader("⚡ Energy Consumption (Training Time)")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(results_df["Model"], results_df["Training Time (s)"], color="#F44336")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Energy Usage Comparison")
    plt.xticks(rotation=20)

    save_and_show(fig, "energy_comparison.png")


# Trade-off Graph
def plot_tradeoff(results_df):
    metric = "R2 Score" if "R2 Score" in results_df.columns else "Accuracy"

    st.subheader("🌱 Accuracy vs Energy Trade-off")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(results_df["Training Time (s)"], results_df[metric], color="#009688")

    for i, model in enumerate(results_df["Model"]):
        ax.text(
            results_df["Training Time (s)"][i],
            results_df[metric][i],
            model
        )

    ax.set_xlabel("Training Time (s)")
    ax.set_ylabel(metric)
    ax.set_title("Accuracy vs Energy")

    save_and_show(fig, "accuracy_vs_energy.png")


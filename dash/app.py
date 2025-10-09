import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


st.set_page_config(page_title="NeuralEdge Bench", layout="wide")
st.title("NeuralEdge Bench Dashboard")

results_path = st.text_input("Results CSV", value="results/summary.csv")
if Path(results_path).exists():
    df = pd.read_csv(results_path)
    st.dataframe(df)
    if {"macro_f1", "p50_ms"}.issubset(df.columns):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.scatterplot(data=df, x="p50_ms", y="macro_f1", hue="variant", ax=ax[0])
        ax[0].set_title("Accuracy vs. Latency")
        if "p90_ms" in df.columns:
            sns.scatterplot(data=df, x="p90_ms", y="macro_f1", hue="variant", ax=ax[1])
            ax[1].set_title("Accuracy vs. P90 Latency")
        st.pyplot(fig)
else:
    st.info("Run bench to populate results.")


import os
import pandas as pd
import streamlit as st
from src.main import run, OUT_DIR  # import OUT_DIR too

st.set_page_config(page_title="SCWE Project", layout="wide")

st.title("🧪 Self-Calibrated Weighted Ensemble (SCWE)")
st.write("""
This app demonstrates **SCWE – Self-Calibrated Weighted Ensemble**,
a novel ML method that calibrates probabilities, assigns validation-based weights,
and tunes thresholds for F1-score optimization.
""")

# Button to run experiment
if st.button("Run SCWE Evaluation"):
    try:
        run()
        st.success("✅ SCWE Evaluation Complete! Check results below.")

        # Show metrics
        results_path = os.path.join(OUT_DIR, "scwe_kfold_results.csv")
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            st.subheader("📊 Results (K-Fold Evaluation)")
            st.dataframe(df)

            st.subheader("📊 Summary Statistics")
            st.text(df.describe().to_string())

        # Show plots
        st.subheader("📈 Visualizations")
        for f in os.listdir(OUT_DIR):
            if f.endswith(".png"):
                st.image(os.path.join(OUT_DIR, f), caption=f)

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")


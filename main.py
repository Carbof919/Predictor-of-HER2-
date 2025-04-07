import streamlit as st
import pandas as pd
import pickle
import os

st.title("💊 Drug Resistance Predictor of HER2+ Breast Cancer")

st.markdown("""
> ⚠️ **Important:** To ensure correct results, your file should contain **gene expression values** for each **cell line** like this:

| CELL_LINE_NAME | GeneA | GeneB | GeneC | ... |
|----------------|-------|-------|-------|-----|
| AU565          | 2.34  | 1.11  | 3.50  | ... |
| SKBR3          | 1.02  | 0.88  | 2.79  | ... |

- The column with cell line names should be labeled something like `CELL_LINE_NAME`, `Cell Line`, or similar.
- Gene columns should match the required features for the selected drug.
""")

# Dropdown with multiple drugs
drug = st.selectbox("Select a drug", [
    "Lapatinib", "Afatinib", "AZD8931",
    "Pelitinib", "CP724714", "Temsirolimus", "Omipalisib"
])

# Optional (but cool): Dynamic subheader based on selected drug
st.subheader(f"🧪 Prediction Results for {drug}")

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV file with gene expression", type=["csv"])
if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.write("👀 Uploaded data preview:")
    st.dataframe(user_df.head())

    try:
        # Load model and feature list
        model_path = f"models/{drug}_model.pkl"
        feature_path = "models/feature_names.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feature_path, "rb") as f:
            features = pickle.load(f)

        # Identify cell line column (flexibly)
        cell_line_col = None
        for col in user_df.columns:
            if "cell" in col.lower() and "line" in col.lower():
                cell_line_col = col
                break

        if not cell_line_col:
            st.error("❌

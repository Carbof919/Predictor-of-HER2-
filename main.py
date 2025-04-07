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
            st.error("❌ Could not find a column with cell line names. Please ensure it contains terms like 'cell' and 'line'.")
            st.stop()

        # Extract cell lines and expression matrix
        CELL_LINE_NAMES = user_df[cell_line_col].tolist()
        expression_data = user_df.drop(columns=[cell_line_col])

        # Match features
        common_genes = [gene for gene in features if gene in expression_data.columns]
        if not common_genes:
            st.error("❌ None of the required genes were found in your uploaded file.")
            st.code("Required genes:\n" + ", ".join(features))
            st.stop()

        # Subset input data
        input_data = expression_data[common_genes]
        preds = model.predict(input_data)

        # Convert predictions to labels
        pred_labels = ["Sensitive" if x == 0 else "Resistant" for x in preds]

        # Create matrix-style DataFrame
        matrix_data = pd.DataFrame([pred_labels] * len(common_genes),  # repeated rows
                                   columns=CELL_LINE_NAMES,
                                   index=common_genes).T  # transpose so cell lines = rows

        st.success("✅ Prediction complete!")
        st.write("🧬 Genes used for prediction:")
        st.code(", ".join(common_genes))

        st.write("📊 **Prediction Matrix** (Cell lines as rows, genes as columns):")
        st.dataframe(matrix_data)

        # 📥 Downloadable version: same as shown matrix
        download_df = matrix_data.copy()
        download_df.insert(0, "CELL_LINE_NAME", download_df.index)
        csv = download_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"{drug}_predictions.csv",
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")



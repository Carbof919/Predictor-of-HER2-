import streamlit as st
import pandas as pd
import pickle
import os

st.title("Drug Resistance Predictor of HER2+ BRCA")
st.write("Upload your gene expression file and select a drug to predict resistance.")

# Dropdown with multiple drugs
drug = st.selectbox("Select a drug", [
    "Lapatinib", "Afatinib", "AZD8931",
    "Pelitinib", "CP724714", "Temsirolimus", "Omipalisib"
])

# File upload
uploaded_file = st.file_uploader("Upload CSV file with gene expression", type=["csv"])

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.dataframe(user_df.head())

    try:
        model_path = f"models/{drug}_model.pkl"
        feature_path = "models/feature_names.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feature_path, "rb") as f:
            features = pickle.load(f)

        # Flexibly identify the column with cell line names
        cell_line_col = None
        for col in user_df.columns:
            if "cell" in col.lower() and "line" in col.lower():
                cell_line_col = col
                break

        if not cell_line_col:
            st.error("⚠️ Please include a column with cell line names (e.g., 'CELL_LINE_NAME', 'cell line', etc.)")
            st.stop()

        CELL_LINE_NAMES = user_df[cell_line_col].tolist()
        expression_data = user_df.drop(columns=[cell_line_col])

        # Match features (genes)
        common_genes = [gene for gene in features if gene in expression_data.columns]
        if not common_genes:
            st.error("❌ None of the required genes are present in the uploaded file.")
        else:
            input_data = expression_data[common_genes]
            preds = model.predict(input_data)
            pred_labels = ["Sensitive" if x == 0 else "Resistant" for x in preds]

            # One row per cell line, one prediction per row
            download_df = pd.DataFrame({
                "Gene": common_genes * len(pred_labels),
                "CELL_LINE_NAME": sum([[name] * len(common_genes) for name in CELL_LINE_NAMES], []),
                "Prediction": sum([[label] * len(common_genes) for label in pred_labels], [])
            })

            st.success("✅ Prediction complete!")
            st.write("🧬 Genes used in prediction:")
            st.code(", ".join(common_genes))

            st.write("📋 **Prediction Table (Cell Lines × Genes):**")
            st.dataframe(download_df)

            csv = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV (Genes × CELL_LINE_NAME with Predictions)",
                data=csv,
                file_name='gene_cellline_predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"🚨 An error occurred: {e}")

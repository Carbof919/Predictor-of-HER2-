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

        # Get cell line names
        if "CELL_LINE_NAMES" in user_df.columns:
            cell_lines = user_df["CELL_LINE_NAMES"].tolist()
            expression_data = user_df.drop(columns=["CELL_LINE_NAMES"])
        else:
            if "CellLine" in user_df.columns:
    cell_lines = user_df["CELL_LINE_NAMESe"].tolist()
    expression_data = user_df.drop(columns=["CELL_LINE_NAMES"])
else:
    st.error("⚠️ Please include a 'CellLine' column in your file with cell line names.")
    st.stop()

            expression_data = user_df.copy()

        # Match features
        common_genes = [gene for gene in features if gene in expression_data.columns]
        if not common_genes:
            st.error("None of the required genes are present in the uploaded file.")
        else:
            input_data = expression_data[common_genes]
            preds = model.predict(input_data)
            pred_labels = ["Sensitive" if x == 0 else "Resistant" for x in preds]

            # Prepare a transposed DataFrame: Genes as rows, CellLines as columns
            result_df = pd.DataFrame([pred_labels], index=["Prediction"], columns=cell_lines)
            result_df.insert(0, "Gene", " / ".join(common_genes))  # Show all used genes

            st.success("Prediction complete!")
            st.write("🧬 Genes used in prediction:")
            st.code(", ".join(common_genes))

            st.write("📋 **Prediction Matrix (Genes vs CELL_LINE_NAMES):**")
            st.dataframe(result_df)

            # Downloadable version (transpose if needed)
            download_df = pd.DataFrame({
                "Gene": common_genes * len(pred_labels),
                "CELL_LINE_NAMES": sum([[name]*len(common_genes) for name in CELL_LINE_NAMES], []),
                "Prediction": sum([[label]*len(common_genes) for label in pred_labels], [])
            })

            csv = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV (Genes × CELL_LINE_NAMES with Predictions)",
                data=csv,
                file_name='gene_cellline_predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

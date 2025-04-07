import streamlit as st
import pandas as pd
import pickle
import os

st.title("Drug Resistance Predictor of HER2+ BRCA")
st.write("Upload your gene expression file and select a drug to predict resistance.")

# 🔹 Format instructions
st.markdown("""
### 📁 File Format Requirements

To ensure accurate predictions, please format your CSV file like this:

- **Rows** = Cell lines  
- **Columns** = Genes  
- One column should clearly indicate **cell line names**, e.g., `CELL_LINE_NAME`, `Cell line`, etc.

| CELL_LINE_NAME | Gene1 | Gene2 | Gene3 | ... |
|----------------|--------|--------|--------|-----|
| AU565          | 1.23   | 0.56   | 2.11   | ... |
| BT474          | 0.87   | 1.02   | 1.45   | ... |

**⚠️ File must be in `.csv` format**
""")

# 🔹 Drug selection
drug = st.selectbox("Select a drug", [
    "Lapatinib", "Afatinib", "AZD8931",
    "Pelitinib", "CP724714", "Temsirolimus", "Omipalisib"
])

# 🔹 File upload
uploaded_file = st.file_uploader("Upload CSV file with gene expression", type=["csv"])
if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.dataframe(user_df.head())

    try:
        # Load model and features
        model_path = f"models/{drug}_model.pkl"
        feature_path = "models/feature_names.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feature_path, "rb") as f:
            features = pickle.load(f)

        # 🔹 Flexibly identify cell line column
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

        # 🔹 Match features
        common_genes = [gene for gene in features if gene in expression_data.columns]
        if not common_genes:
            st.error("None of the required genes are present in the uploaded file.")
        else:
            input_data = expression_data[common_genes]
            preds = model.predict(input_data)
            pred_labels = ["Sensitive" if x == 0 else "Resistant" for x in preds]

            # 🔹 Dynamic subheader
            st.subheader(f"🧪 Prediction Results for {drug}")

            # 🔹 Display results
            result_df = pd.DataFrame({
                "CELL_LINE_NAME": CELL_LINE_NAMES,
                "Prediction": pred_labels
            })
            st.dataframe(result_df)

            # 🔹 Genes used
            st.write("🧬 Genes used in prediction:")
            st.code(", ".join(common_genes))

            # 🔹 Download file
            download_df = pd.DataFrame({
                "Gene": common_genes * len(pred_labels),
                "CELL_LINE_NAME": sum([[name] * len(common_genes) for name in CELL_LINE_NAMES], []),
                "Prediction": sum([[label] * len(common_genes) for label in pred_labels], [])
            })

            csv = download_df.to_csv(index=False).encode('utf-8')
            file_name = f"{drug.lower()}_gene_cellline_predictions.csv"
            st.download_button(
                label="📥 Download CSV (Genes × CELL_LINE_NAME with Predictions)",
                data=csv,
                file_name=file_name,
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

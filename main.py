import streamlit as st
import pandas as pd
import pickle
import os

# Title and description
st.title("Drug Resistance Predictor for HER2+ Breast Cancer")
st.write("Upload your gene expression file and select a drug to predict resistance.")

# 💡 Important file format note
with st.expander("⚠️ Important: File Format Instructions", expanded=True):
    st.markdown("""
    Please format your file like this:

    | Gene1 | Gene2 | Gene3 | ... | CELL_LINE_NAME |
    |-------|-------|-------|-----|----------------|
    |  1.23 |  3.45 | 2.34  | ... | AU565          |
    |  2.12 |  4.33 | 1.67  | ... | BT474          |

    - Column with **cell line names** should be clearly labeled (`CELL_LINE_NAME`, `Cell Line`, etc.)
    - Genes must be in **columns**.
    """)

# Drug selection
drug = st.selectbox("Select a drug", [
    "Lapatinib", "Afatinib", "AZD8931",
    "Pelitinib", "CP724714", "Temsirolimus", "Omipalisib"
])

# Optional dynamic subheader
st.subheader(f"💊 You selected: {drug}")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(user_df.head())

    try:
        # Load model and feature names
        model_path = f"models/{drug}_model.pkl"
        feature_path = "models/feature_names.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feature_path, "rb") as f:
            features = pickle.load(f)

        # Flexibly detect the cell line column
        cell_line_col = None
        for col in user_df.columns:
            if "cell" in col.lower() and "line" in col.lower():
                cell_line_col = col
                break

        if not cell_line_col:
            st.error("⚠️ Please include a column with cell line names (e.g., 'CELL_LINE_NAME').")
            st.stop()

        CELL_LINE_NAMES = user_df[cell_line_col].tolist()
        expression_data = user_df.drop(columns=[cell_line_col])

        # Match features
        common_genes = [gene for gene in features if gene in expression_data.columns]
        if not common_genes:
            st.error("❌ None of the required genes were found in your file.")
            st.stop()

        input_data = expression_data[common_genes]
        preds = model.predict(input_data)
        pred_labels = ["Sensitive" if x == 0 else "Resistant" for x in preds]

        # ✅ Show prediction matrix: cell lines (rows) × genes (columns)
        matrix_data = pd.DataFrame(
            data=[[label] * len(common_genes) for label in pred_labels],
            index=CELL_LINE_NAMES,
            columns=common_genes
        )

        st.success("✅ Prediction complete!")
        st.write("🧬 Genes used in prediction:")
        st.code(", ".join(common_genes))

        st.subheader(f"🧪 Prediction Matrix for {drug}")
        st.dataframe(matrix_data)

# 📥 Downloadable version: same as shown matrix (cell lines as rows, genes as columns)
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
        st.error(f"❗ An error occurred: {e}")

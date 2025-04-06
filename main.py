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

        # Match features
        common_genes = [gene for gene in features if gene in user_df.columns]
        if not common_genes:
            st.error("None of the required genes are present in the uploaded file.")
        else:
            input_data = user_df[common_genes]

            # Get sample/cell line names
            if "CELL_LINE_NAMES" in user_df.columns:
                sample_names = user_df["CELL_LINE_NAMES"]
            else:
                sample_names = [f"Sample_{i}" for i in range(len(user_df))]

            preds = model.predict(input_data)
            pred_labels = ["Sensitive" if x == 0 else "Resistant" for x in preds]

            # Combine results
            result_df = pd.DataFrame({
                "CELL_LINE_NAMES": sample_names,
                "Prediction": pred_labels
            })

            st.success("Prediction complete!")
            st.write("🧬 Genes used in prediction:")
            st.code(", ".join(common_genes))

            st.write("📋 **Prediction Result Table** (with Cell Line Names):")
            st.dataframe(result_df)

            # Downloadable CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV with predictions and CELL_LINE_NAMES",
                data=csv,
                file_name='CELL_LINE_NAMES_predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

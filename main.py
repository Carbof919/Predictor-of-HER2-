import streamlit as st
import pandas as pd
import pickle
import os

st.title("Drug Resistance Predictor")
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
            preds = model.predict(input_data)
            pred_labels = ["Sensitive" if x == 0 else "Resistant" for x in preds]

            # Show genes used
            st.write("🧬 Genes used in prediction:")
            st.code(", ".join(common_genes))

            # Create a result DataFrame (new, doesn't modify original)
            result_df = input_data.copy()
            result_df["Prediction"] = pred_labels
            result_df.index.name = "Sample"

            st.success("Prediction complete!")
            st.write("### 📋 Predictions with Gene Values")
            st.dataframe(result_df)

            # Downloadable CSV
            csv = result_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='predictions_with_genes.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")

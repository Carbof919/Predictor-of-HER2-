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
            # 👇 New: show genes used for prediction
            st.write("🧬 Genes used in prediction:")
            st.write(common_genes)

            input_data = user_df[common_genes]
            preds = model.predict(input_data)
            st.success("Prediction complete!")

            # 👇 New: display results as labeled dataframe
            result_df = pd.DataFrame({
                "Sample ID": user_df.index,
                "Prediction": ["Sensitive" if p == 1 else "Resistant" for p in preds]
            })
            st.write(result_df)

            # 👇 New: download button for results
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
            
            # 👇 Original: raw predictions as list
            st.write("Raw Predictions:", preds.tolist())

    except Exception as e:
        st.error(f"An error occurred: {e}")

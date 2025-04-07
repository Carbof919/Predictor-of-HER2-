import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Multi-Drug Response Predictor", layout="wide")

st.title("💊 Drug Response Predictor")
st.markdown("""
Upload your gene expression data and select multiple drugs to predict response (Sensitive or Resistant).  
**Note:** One shared `feature.pkl` is used across all models.
""")

# 📄 Show example input format
with st.expander("📄 Example of Required Input Format"):
    sample_df = pd.DataFrame({
        "Gene1": [7.2, 5.1],
        "Gene2": [3.3, 6.7],
        "Gene3": [4.9, 2.8]
    }, index=["CellLine1", "CellLine2"])
    st.dataframe(sample_df)

# Upload expression file
uploaded_file = st.file_uploader("📤 Upload Gene Expression CSV", type=["csv"])

# Load common features
try:
    feature_genes = joblib.load("feature_names.pkl")  # Common feature set
except Exception as e:
    st.error("❌ Failed to load feature.pkl. Make sure the file exists.")
    st.stop()

# Load available models
model_dir = "models"
available_drugs = [f.replace("_model.pkl", "") for f in os.listdir(model_dir) if f.endswith("_model.pkl")]

# Select drugs
selected_drugs = st.multiselect("🧪 Select Drugs to Predict", available_drugs)

# Prediction
if uploaded_file and selected_drugs:
    df_input = pd.read_csv(uploaded_file, index_col=0)

    try:
        df_filtered = df_input[feature_genes]
    except KeyError as e:
        missing = list(set(feature_genes) - set(df_input.columns))
        st.error(f"❌ Missing required genes in input file: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        st.stop()

    result_df = df_input.copy()

    for drug in selected_drugs:
        model_path = os.path.join(model_dir, f"{drug}_model.pkl")
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Model for {drug} not found. Skipping.")
            continue

        model = joblib.load(model_path)
        preds = model.predict(df_filtered)
        result_df[f"{drug}_Response"] = preds

    # Show predictions
    st.subheader("🧾 Combined Prediction Result")
    st.dataframe(result_df)

    # Download CSV
    st.download_button(
        label="📥 Download Prediction CSV",
        data=result_df.to_csv().encode('utf-8'),
        file_name="drug_response_predictions.csv",
        mime="text/csv"
    )

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Drug Response Predictor", layout="wide")

st.title("💊 Multi-Drug Response Prediction App")
st.markdown("""
Welcome to the **Drug Response Predictor App**!  
Upload your gene expression file and choose one or more drugs to see the predicted response (Sensitive or Resistant).

#### ⚠️ Important:
- Format your input file as: **Rows = Cell Lines**, **Columns = Gene Symbols**
- File must be in **CSV** format with cell line names as row indexes.

---
""")

# Sample format display
sample_df = pd.DataFrame({
    "Gene1": [5.6, 7.8],
    "Gene2": [2.3, 6.1],
    "Gene3": [4.9, 3.5]
}, index=["CellLine1", "CellLine2"])
with st.expander("📄 See Input File Format Example"):
    st.dataframe(sample_df)

# File upload
uploaded_file = st.file_uploader("📤 Upload your Gene Expression CSV", type=["csv"])

# Drug selection
model_dir = "models"
feature_dir = "features"
available_drugs = [f.replace("_model.pkl", "") for f in os.listdir(model_dir) if f.endswith("_model.pkl")]
selected_drugs = st.multiselect("🧪 Choose One or More Drugs", available_drugs)

# Perform prediction
if uploaded_file is not None and selected_drugs:
    df_input = pd.read_csv(uploaded_file, index_col=0)

    for drug in selected_drugs:
        st.subheader(f"🧪 Prediction Results for {drug}")

        # Load features and model
        feature_path = os.path.join(feature_dir, f"{drug}_features.pkl")
        model_path = os.path.join(model_dir, f"{drug}_model.pkl")

        if not os.path.exists(feature_path) or not os.path.exists(model_path):
            st.warning(f"⚠️ Missing model or feature file for {drug}. Skipping.")
            continue

        features = joblib.load(feature_path)
        model = joblib.load(model_path)

        try:
            df_filtered = df_input[features]
        except KeyError:
            st.error(f"🚫 One or more required genes for {drug} not found in your input file.")
            continue

        predictions = model.predict(df_filtered)
        result_df = pd.DataFrame(predictions, index=df_input.index, columns=[f"{drug}_Response"])
        preview_df = pd.concat([df_input, result_df], axis=1)

        # Show results
        st.dataframe(preview_df)

        # Download button
        csv = preview_df.to_csv().encode('utf-8')
        st.download_button(
            label=f"📥 Download Results for {drug}",
            data=csv,
            file_name=f"{drug}_predictions.csv",
            mime='text/csv'
        )

import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Drug Resistance Predictor", layout="wide")
st.title("🧬 Drug Resistance Predictor")
st.markdown("---")
st.write("Upload your gene expression file(s) and select a drug to predict resistance.")

# Select a drug
drug = st.selectbox("💊 Select a drug", [
    "Lapatinib", "Afatinib", "AZD8931",
    "Pelitinib", "CP724714", "Temsirolimus", "Omipalisib"
])

# Upload multiple files
uploaded_files = st.file_uploader(
    "📂 Upload one or more CSV files with gene expression data",
    type=["csv"],
    accept_multiple_files=True
)

# Load model and feature names
try:
    model_path = f"models/{drug}_model.pkl"
    feature_path = "models/feature_names.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feature_path, "rb") as f:
        features = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model or features: {e}")
    st.stop()

# Process each uploaded file
for uploaded_file in uploaded_files:
    st.markdown("---")
    st.subheader(f"📄 File: {uploaded_file.name}")

    user_df = pd.read_csv(uploaded_file)
    st.write("🔍 Preview of uploaded data:")
    st.dataframe(user_df.head())

    # Match features
    common_genes = [gene for gene in features if gene in user_df.columns]
    if not common_genes:
        st.error("⚠️ None of the required genes are present in the uploaded file.")
        continue

    input_data = user_df[common_genes]

    try:
        preds = model.predict(input_data)
        pred_labels = ["Sensitive" if x == 0 else "Resistant" for x in preds]

        # Show genes used
        st.write("🧬 Genes used in prediction:")
        st.code(", ".join(common_genes))

        # Prepare readable output per sample
        st.write("### 📊 Sample-wise Prediction Summary")
        for idx, row in input_data.iterrows():
            st.markdown(f"**Sample {idx + 1}: {pred_labels[idx]}**")
            sample_genes = row.to_dict()
            gene_list_text = "\n".join([f"- `{gene}`: {val}" for gene, val in sample_genes.items()])
            st.markdown(gene_list_text)

        # Bar Chart
        st.write("### 📈 Overall Prediction Summary")
        counts = pd.Series(preds).value_counts().sort_index()
        labels = ["Sensitive (0)", "Resistant (1)"]
        values = [counts.get(0, 0), counts.get(1, 0)]

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=["#4CAF50", "#F44336"])
        ax.set_ylabel("Number of Samples")
        ax.set_title("Resistance Prediction Distribution")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

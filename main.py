import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestClassifier

# Load the common feature list
with open("feature.pkl", "rb") as f:
    feature_genes = pickle.load(f)

# Title & upload
st.title("🔬 Multi-Drug Resistance Predictor")

st.markdown("""
**ℹ️ Important Note:** Format your CSV like this:
- Rows = cell lines  
- Columns = genes (at least the required ones)

You must have expression data for required features.
""")

uploaded_file = st.file_uploader("Upload your gene expression CSV file", type=["csv"])

# Allow selecting multiple drugs
selected_drugs = st.multiselect("Select Drug(s)", os.listdir("models"))

# Predict button
if uploaded_file and selected_drugs and st.button("Run Prediction"):
    user_data = pd.read_csv(uploaded_file, index_col=0)
    gene_input = user_data.copy()

    # Only include columns that are in the model
    missing_genes = [g for g in feature_genes if g not in gene_input.columns]
    if missing_genes:
        st.warning(f"⚠️ Missing genes in input: {missing_genes[:5]}... ({len(missing_genes)} total)")
    available_genes = [g for g in feature_genes if g in gene_input.columns]
    gene_input = gene_input[available_genes]

    results = pd.DataFrame(index=user_data.index)
    results["Cell Line"] = results.index

    for drug_file in selected_drugs:
        drug_name = os.path.splitext(drug_file)[0]
        with open(f"models/{drug_file}", "rb") as f:
            model = pickle.load(f)

        pred = model.predict(gene_input)
        pred_labels = ["Resistant" if p == 0 else "Sensitive" for p in pred]
        results[f"{drug_name}_Response"] = pred_labels

    # Add the selected features back
    result_display = user_data[available_genes].copy()
    result_display.insert(0, "Cell Line", user_data.index)

    # Append all drug response columns after Cell Line
    for drug_file in selected_drugs:
        drug_name = os.path.splitext(drug_file)[0]
        result_display[f"{drug_name}_Response"] = results[f"{drug_name}_Response"].values

    # Preview
    st.subheader("🧪 Prediction Results")
    st.write(result_display)

    # Download
    drug_names = "_".join([os.path.splitext(drug)[0] for drug in selected_drugs])
    csv = result_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name=f"{drug_names}_Predictions.csv",
        mime="text/csv"
    )

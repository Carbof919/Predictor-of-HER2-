import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestClassifier

# Load the common feature list
with open("feature_names.pkl", "rb") as f:
    feature_genes = pickle.load(f)

# Title & upload
st.title("🔬 Multi-Drug Resistance Predictor")

st.markdown("""
> ⚠️ **Important:** To ensure correct results, your file should contain **gene expression values** for each **cell line** like this:

| CELL_LINE_NAME | GeneA | GeneB | GeneC | ... |
|----------------|-------|-------|-------|-----|
| AU565          | 2.34  | 1.11  | 3.50  | ... |
| SKBR3          | 1.02  | 0.88  | 2.79  | ... |

- The column with cell line names should be labeled something like `CELL_LINE_NAME`, `Cell Line`, or similar.
- Gene columns should match the required features for the selected drug.
""")

uploaded_file = st.file_uploader("Upload your gene expression CSV file", type=["csv"])

# Load model filenames (exclude feature_names.pkl)
model_folder = "models"
model_files = [f for f in os.listdir(model_folder) if f.endswith(".pkl") and f != "feature_names.pkl"]

# Map clean drug name -> filename
drug_name_map = {os.path.splitext(f)[0]: f for f in model_files}

# Drug selector (clean names only)
selected_clean_names = st.multiselect("Select Drug(s)", list(drug_name_map.keys()))

# Predict
if uploaded_file and selected_clean_names and st.button("Run Prediction"):
    user_data = pd.read_csv(uploaded_file, index_col=0)
    gene_input = user_data.copy()

    # Make sure only available genes are used for prediction
    available_genes = [g for g in feature_genes if g in gene_input.columns]
    missing_genes = [g for g in feature_genes if g not in gene_input.columns]
    if missing_genes:
        st.warning(f"⚠️ Missing genes in input: {missing_genes[:5]}... ({len(missing_genes)} total)")
    gene_input = gene_input[available_genes]

    # Prepare results
    results = pd.DataFrame(index=user_data.index)
    results["Cell Line"] = results.index

    for drug_name in selected_clean_names:
        model_file = drug_name_map[drug_name]
        with open(f"{model_folder}/{model_file}", "rb") as f:
            model = pickle.load(f)

        pred = model.predict(gene_input)
        pred_labels = ["Resistant" if p == 0 else "Sensitive" for p in pred]
        results[f"{drug_name}_Response"] = pred_labels

    # Prepare final output: full user data + predictions after Cell Line
    full_data = user_data.copy()
    full_data.insert(0, "Cell Line", full_data.index)

    for drug_name in selected_clean_names:
        full_data[f"{drug_name}_Response"] = results[f"{drug_name}_Response"].values

    # Show result
    st.subheader("🧪 Prediction Results")
    st.write(full_data)

    # Download
    drug_names_joined = "_".join(selected_clean_names)
    csv = full_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name=f"{drug_names_joined}_Predictions.csv",
        mime="text/csv"
    )

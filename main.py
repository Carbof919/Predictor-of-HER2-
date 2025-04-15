import streamlit as st
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="🧠 MultiDrugIntel", page_icon="🧪", layout="wide")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.image("Image.png", width=120)
st.sidebar.title("🧬 MultiDrugIntel")
st.sidebar.markdown("""
**A ML-powered app to predict drug resistance in breast cancer.**

📌 Modes:
- Upload gene expression data
- Manual gene entry

📊 Visualize IC50 profiles
🔬 Predict resistance for 19 drugs
""")

# -----------------------
# Title & Tabs
# -----------------------
st.title("🧠 MultiDrugIntel - Multi-Drug Resistance Predictor")
tabs = st.tabs(["🏠 Home", "📊 Visualize", "🔬 Predict"])

# -----------------------
# Load required files
# -----------------------
with open("feature_names.pkl", "rb") as f:
    feature_genes = pickle.load(f)

with open("drug_ic50_data.json", "r") as f:
    drug_ic50_data = json.load(f)

# -----------------------
# Drug model map
# -----------------------
drug_name_map = {
    "AST-1306": "AST-1306_model.pkl",
    "Axitinib": "Axitinib_model.pkl",
    "AZD4547": "AZD4547_model.pkl",
    "Bicalutamide": "Bicalutamide_model.pkl",
    "BMS-754807": "BMS-754807_model.pkl",
    "Cetuximab": "Cetuximab_model.pkl",
    "CP724714": "CP724714_model.pkl",
    "CUDC-101": "CUDC-101_model.pkl",
    "Dactosilib": "Dactosilib_model.pkl",
    "GSK690693": "GSK690693_model.pkl",
    "Panobinostat": "Panobinostat_model.pkl",
    "Selumetinib": "Selumetinib_model.pkl",
    "Tivozanib": "Tivozanib_model.pkl",
    "Lapatinib": "Lapatinib_model.pkl",
    "Afatinib": "Afatinib_model.pkl",
    "AZD8931": "AZD8931_model.pkl",
    "Pelitinib": "Pelitinib_model.pkl",
    "Temsirolimus": "Temsirolimus_model.pkl",
    "Omipalisib": "Omipalisib_model.pkl"
}

# -----------------------
# 🏠 Home Tab
# -----------------------
with tabs[0]:
    st.subheader("📚 Instructions")
    st.markdown("""
    **Choose one of the modes below:**

    🔹 **Mode 1:** Upload gene expression CSV with cell line names + gene symbols.

    🔹 **Mode 2:** Upload gene expression CSV with gene symbols only (no cell lines).

    🔹 **Mode 3:** Manually input gene names + expression values.

    ⚠️ Ensure gene names match those in the trained models.

    ---

    ### 📁 Data Format Examples:

    #### 🔹 Mode 1: Expression + Cell Lines
    | CELL_LINE_NAME | GeneA | GeneB | GeneC | ... |
    |----------------|--------|--------|--------|-----|
    | AU565          | 2.34   | 1.11   | 3.50   | ... |
    | SKBR3          | 1.02   | 0.88   | 2.79   | ... |

    - The **first column** should contain cell line names.
    - Gene columns should match the required features for the selected drug(s).

    #### 🔹 Mode 2: Expression Only (No Cell Line Column)
    | GeneA | GeneB | GeneC | ... |
    |--------|--------|--------|-----|
    | 2.34   | 1.11   | 3.50   | ... |
    | 1.02   | 0.88   | 2.79   | ... |
    """)

# -----------------------
# 📊 Visualize Tab
# -----------------------
with tabs[1]:
    st.subheader("📊 Drug Sensitivity and IC50 Visualization")

    selected_drug = st.selectbox("💊 Choose a drug to visualize", list(drug_ic50_data.keys()))

    drug_dict = drug_ic50_data.get(selected_drug, {})
    if isinstance(drug_dict, dict):
        data = pd.DataFrame(list(drug_dict.items()), columns=["CELL_LINE_NAME", "LN_IC50"])
        threshold = data["LN_IC50"].median()
        data["Label"] = data["LN_IC50"].apply(lambda x: "Resistant" if x > threshold else "Sensitive")

        fig, ax = plt.subplots(1, 2, figsize=(16, 5))
        sns.countplot(data=data, x="Label", ax=ax[0], palette="Set2")
        ax[0].set_title(f"Resistance Distribution for {selected_drug}")

        sns.barplot(data=data, x="CELL_LINE_NAME", y="LN_IC50", ax=ax[1], palette="viridis")
        ax[1].set_title(f"Log(IC50) values for {selected_drug}")
        ax[1].tick_params(axis='x', rotation=90)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No valid IC50 data available for this drug.")

# -----------------------
# 🔬 Predict Tab
# -----------------------
with tabs[2]:
    st.subheader("🔬 Predict Drug Response")

    mode = st.radio("Select Input Mode:", [
        "Mode 1: Cell line + Expression",
        "Mode 2: Expression only",
        "Mode 3: Manual input"
    ])
    selected_drugs = st.multiselect("💊 Select Drug(s) to Predict", list(drug_name_map.keys()))

    def load_model(drug):
        try:
            model_path = os.path.join("models", f"{drug}_model.pkl")
            return pickle.load(open(model_path, "rb"))
        except Exception as e:
            st.error(f"❌ Error loading model for {drug}: {e}")
            return None

    def align_features(input_df, model):
        model_features = model.get_booster().feature_names
        input_df = input_df.reindex(columns=model_features, fill_value=0)
        return input_df

    if mode == "Mode 1: Cell line + Expression":
        uploaded_file = st.file_uploader("📁 Upload your gene expression CSV file", type=["csv"])
        if uploaded_file and selected_drugs:
            user_data = pd.read_csv(uploaded_file, index_col=0)
            gene_input = user_data.copy()

            results = pd.DataFrame(index=user_data.index)
            results["Cell Line"] = results.index

            for drug in selected_drugs:
                model = load_model(drug)
                if model:
                    gene_input_aligned = align_features(gene_input.copy(), model)
                    pred = model.predict(gene_input_aligned)
                    pred_labels = ["Resistant" if p == 0 else "Sensitive" for p in pred]
                    results[f"{drug}_Response"] = pred_labels

            full_data = user_data.copy()
            full_data.insert(0, "Cell Line", full_data.index)
            for drug in selected_drugs:
                full_data[f"{drug}_Response"] = results[f"{drug}_Response"].values

            st.write(full_data)

            filename = "_".join(selected_drugs) + "_Predictions.csv"
            st.download_button(
                label="📅 Download Predictions",
                data=full_data.to_csv(index=False).encode("utf-8"),
                file_name=filename,
                mime="text/csv"
            )

    elif mode == "Mode 2: Expression only":
        exp_file = st.file_uploader("📁 Upload CSV with expression values only", type=["csv"])
        if exp_file and selected_drugs:
            gene_input = pd.read_csv(exp_file)
            results = {}
            for drug in selected_drugs:
                model = load_model(drug)
                if model:
                    gene_input_aligned = align_features(gene_input.copy(), model)
                    pred = model.predict(gene_input_aligned)
                    pred_labels = ["Resistant" if p == 0 else "Sensitive" for p in pred]
                    results[drug] = pred_labels
            st.write(pd.DataFrame(results))

    elif mode == "Mode 3: Manual input":
        st.write("✍️ Input gene expressions")
        gene_input = {}
        for gene in feature_genes[:10]:
            gene_input[gene] = st.number_input(f"{gene}", value=1.0)
        input_df = pd.DataFrame([gene_input])
        if selected_drugs:
            results = {}
            for drug in selected_drugs:
                model = load_model(drug)
                if model:
                    input_df_aligned = align_features(input_df.copy(), model)
                    pred = model.predict(input_df_aligned)
                    results[drug] = "Resistant" if pred[0] == 0 else "Sensitive"
            st.write("📋 Prediction Results:")
            st.write(pd.DataFrame([results]))

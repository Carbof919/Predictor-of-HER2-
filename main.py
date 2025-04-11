import streamlit as st
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

# Load the common feature list
with open("feature_names.pkl", "rb") as f:
    feature_genes = pickle.load(f)

# Load IC50 Data
@st.cache_data

def load_ic50_data():
    with open("drug_ic50_data.json", "r") as f:
        return json.load(f)

ic50_data = load_ic50_data()

# App title
st.set_page_config(page_title="MultiDrugIntel", layout="wide")
st.title("🧠 MultiDrugIntel - Multi-Drug Resistance Predictor")

# Tabs
selected_tab = st.sidebar.radio("Navigation", [
    "Predict Drug Resistance",
    "Visualize IC50"
])

if selected_tab == "Predict Drug Resistance":
    st.markdown("""
    > ⚠️ **Important:** Upload gene expression data like below:

    | CELL_LINE_NAME | GeneA | GeneB | GeneC | ... |
    |----------------|-------|-------|-------|-----|
    | AU565          | 2.34  | 1.11  | 3.50  | ... |
    | SKBR3          | 1.02  | 0.88  | 2.79  | ... |

    - The **first column** should contain cell line names.
    - Gene columns should match the required features for the selected drug(s).
    """)

    uploaded_file = st.file_uploader("📁 Upload your gene expression CSV file", type=["csv"])

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


    selected_drugs = st.multiselect("💊 Select Drug(s) to Predict Response", list(drug_name_map.keys()))

    if uploaded_file and selected_drugs and st.button("Run Prediction"):
        user_data = pd.read_csv(uploaded_file, index_col=0)
        gene_input = user_data.copy()

        available_genes = [g for g in feature_genes if g in gene_input.columns]
        missing_genes = [g for g in feature_genes if g not in gene_input.columns]

        if missing_genes:
            st.warning(f"⚠️ Missing genes: {missing_genes[:5]}... ({len(missing_genes)} total)")
        gene_input = gene_input[available_genes]

        results = pd.DataFrame(index=user_data.index)
        results["Cell Line"] = results.index

        for drug in selected_drugs:
            model_file = drug_name_map[drug]
            with open(os.path.join("models", model_file), "rb") as f:
                model = pickle.load(f)

            pred = model.predict(gene_input)
            pred_labels = ["Resistant" if p == 0 else "Sensitive" for p in pred]
            results[f"{drug}_Response"] = pred_labels

        full_data = user_data.copy()
        full_data.insert(0, "Cell Line", full_data.index)

        for drug in selected_drugs:
            full_data[f"{drug}_Response"] = results[f"{drug}_Response"].values

        st.subheader("🧪 Prediction Results")
        st.write(full_data)

        filename = "_".join(selected_drugs) + "_Predictions.csv"
        st.download_button(
            label="📥 Download Predictions",
            data=full_data.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv"
        )

elif selected_tab == "Visualize IC50":
    st.header("💊 IC50 Concentration Visualizer")
    st.markdown("""
    This section lets you explore **IC50 drug concentrations** across different cell lines.

    - **Lower IC50** = Higher potency
    - Use this for **biological validation** alongside ML predictions.
    """)

    selected_drugs = st.multiselect("Select Drug(s) to visualize IC50 levels", sorted(list(ic50_data.keys())))

    if selected_drugs:
        for drug in selected_drugs:
            st.subheader(f"📈 {drug} IC50 Concentration")

            data = ic50_data[drug]
            df = pd.DataFrame(list(data.items()), columns=["Cell Line", "IC50"])
            df = df.sort_values(by="IC50")

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=df, x="Cell Line", y="IC50", palette="coolwarm", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)

            with st.expander(f"🔬 Show raw IC50 values for {drug}"):
                st.dataframe(df)
    else:
        st.info("📌 Select one or more drugs to see IC50 concentration charts.")

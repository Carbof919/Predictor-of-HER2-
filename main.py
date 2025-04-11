import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier

# Load common feature list
with open("feature_names.pkl", "rb") as f:
    feature_genes = pickle.load(f)

# App title
st.set_page_config(page_title="MultiDrugIntel", layout="wide")
st.title("🧠 MultiDrugIntel - Multi-Drug Resistance Predictor")

# Tabs
tabs = st.tabs(["🏠 Home", "🧪 Run Prediction", "📊 Visualize"])

# --- Home ---
with tabs[0]:
    st.markdown("""
    ## Welcome to MultiDrugIntel
    
    🎯 A machine learning-powered tool to predict drug resistance in cancer based on gene expression.

    ### Features:
    - Upload gene expression datasets
    - Predict resistance to multiple cancer drugs
    - Visualize drug sensitivity outcomes

    > 💡 Built with 💻 Python, 🤖 scikit-learn, 🎨 Streamlit, and 🔬 multi-omics research
    """)

# --- Run Prediction ---
with tabs[1]:
    st.subheader("Choose Input Mode for Prediction")
    mode = st.radio("Select Input Type:", [
        "🧬 Mode 1: CSV with Cell Line + Expression Data",
        "🧬 Mode 2: CSV with Expression Only",
        "📝 Mode 3: Enter Gene Names & Expression Manually"
    ])

    drug_name_map ={
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

    selected_drugs = st.multiselect("💊 Select Drug(s)", list(drug_name_map.keys()))
    prediction_results = None

    if mode == "🧬 Mode 1: CSV with Cell Line + Expression Data":
        uploaded_file = st.file_uploader("📁 Upload CSV (Index = Cell Line, Columns = Genes)", type=["csv"])
        if uploaded_file and selected_drugs and st.button("🚀 Predict"):
            user_data = pd.read_csv(uploaded_file, index_col=0)
            available_genes = [g for g in feature_genes if g in user_data.columns]
            gene_input = user_data[available_genes]
            results = pd.DataFrame(index=user_data.index)
            results["Cell Line"] = results.index
            for drug in selected_drugs:
                with open(os.path.join("models", drug_name_map[drug]), "rb") as f:
                    model = pickle.load(f)
                pred = model.predict(gene_input)
                results[f"{drug}_Response"] = ["Resistant" if p == 0 else "Sensitive" for p in pred]
            st.success("✅ Prediction Complete")
            st.dataframe(results)
            prediction_results = results
            st.download_button("📥 Download", data=results.to_csv().encode(), file_name="Predictions.csv")

    elif mode == "🧬 Mode 2: CSV with Expression Only":
        uploaded_file = st.file_uploader("📁 Upload CSV (Only Gene Expression Values)", type=["csv"])
        if uploaded_file and selected_drugs and st.button("🚀 Predict"):
            user_data = pd.read_csv(uploaded_file)
            available_genes = [g for g in feature_genes if g in user_data.columns]
            gene_input = user_data[available_genes]
            results = pd.DataFrame()
            for drug in selected_drugs:
                with open(os.path.join("models", drug_name_map[drug]), "rb") as f:
                    model = pickle.load(f)
                pred = model.predict(gene_input)
                results[f"{drug}_Response"] = ["Resistant" if p == 0 else "Sensitive" for p in pred]
            st.success("✅ Prediction Complete")
            st.dataframe(results)
            prediction_results = results
            st.download_button("📥 Download", data=results.to_csv().encode(), file_name="Predictions.csv")

    elif mode == "📝 Mode 3: Enter Gene Names & Expression Manually":
        st.markdown("📌 Enter expression values for known genes below.")
        gene_inputs = {}
        for gene in feature_genes[:10]:
            gene_inputs[gene] = st.number_input(f"{gene}", min_value=0.0, max_value=10000.0, step=0.01)
        if selected_drugs and st.button("🚀 Predict"):
            sample_input = np.array([gene_inputs.get(g, 0) for g in feature_genes]).reshape(1, -1)
            results = {}
            for drug in selected_drugs:
                with open(os.path.join("models", drug_name_map[drug]), "rb") as f:
                    model = pickle.load(f)
                pred = model.predict(sample_input)[0]
                results[drug] = "Resistant" if pred == 0 else "Sensitive"
            prediction_results = pd.DataFrame([results])
            st.success("✅ Prediction Complete")
            st.dataframe(prediction_results)

# --- Visualize ---
with tabs[2]:
    st.subheader("📊 Drug Response Visualization")
    if prediction_results is not None:
        for drug in selected_drugs:
            if f"{drug}_Response" in prediction_results.columns:
                fig = px.histogram(prediction_results, x=f"{drug}_Response", color=f"{drug}_Response",
                                   title=f"{drug} Response Distribution",
                                   labels={f"{drug}_Response": "Response"})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Run a prediction first to view visualization.")

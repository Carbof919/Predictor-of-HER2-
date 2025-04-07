import streamlit as st
import pandas as pd
import joblib
import base64

st.set_page_config(page_title="Drug Response Predictor", layout="wide")
st.title("🧬 Drug Response Prediction App")

# Instructions
st.markdown("""
### 📂 Upload Your Gene Expression File
- CSV format
- Rows: Cell lines, Columns: Genes
- Example format:
""")

example_data = pd.DataFrame({
    "Gene1": [2.3, 1.1, 0.5],
    "Gene2": [0.4, 3.2, 2.1],
    "Gene3": [1.2, 1.0, 0.3]
}, index=["CellLine1", "CellLine2", "CellLine3"])
st.dataframe(example_data)

# Upload input data
uploaded_file = st.file_uploader("📁 Upload Gene Expression CSV", type=["csv"])
input_df = None
if uploaded_file:
    input_df = pd.read_csv(uploaded_file, index_col=0)
    st.success("✅ File uploaded and processed successfully!")
    st.write("### 👀 Preview of Uploaded Data")
    st.dataframe(input_df.head())

# Load list of drugs
try:
    with open("models/drug_list.txt") as f:
        drug_list = f.read().splitlines()
except:
    drug_list = []

st.markdown("---")
st.header("🔬 Predict Single Drug Response")
drug = st.selectbox("💊 Select a drug", drug_list)

if st.button("🚀 Predict Response"):
    if input_df is not None:
        try:
            model = joblib.load(f"models/model_{drug}.pkl")
            features = joblib.load(f"features/features_{drug}.pkl")
            filtered_input = input_df[features]
            pred = model.predict(filtered_input)
            result_df = pd.DataFrame({
                "Cell Line": input_df.index,
                "Prediction": pred
            })
            result_df = pd.concat([result_df, filtered_input.reset_index(drop=True)], axis=1)

            st.subheader(f"🧪 Prediction Results for {drug}")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📅 Download Results",
                data=csv,
                file_name=f"{drug}_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please upload a gene expression file first.")

st.markdown("---")
st.header("🧪 Multi-Drug Response Prediction")

selected_drugs = st.multiselect("📌 Select up to 3 drugs", drug_list, max_selections=3)

if st.button("🚀 Run Multi-Drug Prediction"):
    if input_df is not None and selected_drugs:
        results = pd.DataFrame()
        results["Cell Line"] = input_df.index

        for d in selected_drugs:
            try:
                model = joblib.load(f"models/model_{d}.pkl")
                genes = joblib.load(f"features/features_{d}.pkl")
                filtered_input = input_df[genes]
                pred = model.predict(filtered_input)
                results[d] = pred
            except Exception as e:
                st.warning(f"Prediction failed for {d}: {e}")

        st.subheader("📋 Multi-Drug Prediction Results")
        st.dataframe(results)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📅 Download Multi-Drug Predictions",
            csv,
            "multi_drug_predictions.csv",
            "text/csv"
        )
    else:
        st.warning("Please upload a gene expression file and select drugs.")

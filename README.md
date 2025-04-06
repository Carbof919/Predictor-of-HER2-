# Predictor-of-HER2-BRCA


 🧬 Drug Resistance Predictor of HER2+ Breast Cancer (HER2+ BRCA)

This Streamlit application predicts drug resistance in HER2-positive breast cancer cell lines based on uploaded gene expression data. It supports multiple drugs and provides an easy-to-use interface for researchers or clinicians.

 🚀 Features

 ✅ Supports multiple targeted drugs (e.g., Lapatinib, Afatinib, AZD8931, etc.)
 📁 Accepts gene expression data in CSV format
 🧠 Loads pretrained machine learning models
 🔍 Automatically detects the "cell line name" column (flexible naming)
 📊 Predicts drug resistance (Sensitive/Resistant) per cell line
 📥 Allows download of predictions with gene expression profiles



 📦 Folder Structure

project_root/
├── app.py                 # Main Streamlit app
├── models/
│   ├── Lapatinib_model.pkl
│   ├── Afatinib_model.pkl
│   ├── ... (other drug models)
│   └── feature_names.pkl  # List of gene features used for training
└── README.md              # This file




 📄 Input File Format
 
- Accepted format: `.csv`
- Required:
  - One column that contains **cell line names** (name can be flexible like `CELL_LINE_NAME`, `cell line`, `CellLine`, etc.)
  - Other columns: **Gene expression values**, one gene per column

 Example:

| CELL_LINE_NAME | TP53  | ERBB2 | MYC   | ... |
|----------------|-------|-------|-------|-----|
| AU565          | 4.2   | 5.1   | 3.9   | ... |
| BT474          | 3.8   | 4.9   | 2.7   | ... |



🧪 How to Run the App

 ✅ Requirements

- Python 3.7+
- Streamlit
- Pandas
- Pickle (standard lib)

🔧 Installation

bash
 1. Clone the repo
git clone https://github.com/your-username/her2-drug-resistance-predictor.git
cd her2-drug-resistance-predictor

 2. Install dependencies
pip install -r requirements.txt
```

 ▶️ Run the App

bash
streamlit run app.py




📤 Output

- A preview of predictions:
  - Rows = Cell lines
  - Columns = `CELL_LINE_NAME`, `Prediction`, followed by gene expression values (optional)
- Option to download the result as a CSV.



 💡 Notes

- The app tries to auto-detect the cell line column. Ensure the name contains both `"cell"` and `"line"` (case-insensitive).
- If your file doesn’t contain required genes used in the model, a warning will be shown.
- Models must be stored inside the `models/` folder with filenames like: `DrugName_model.pkl`.



 👨‍🔬 Credits

This app was created as part of a research project focused on Predicting drug resistance in HER2+ breast cancer using machine learning and multi-omics data.

---



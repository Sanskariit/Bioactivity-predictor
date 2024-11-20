import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import base64
import joblib

# Load trained model
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please ensure `bioactivity_model.pkl` is in the same directory.")
    st.stop()

# Function to create a download link for results
def create_download_link(df, filename="prediction_output.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'

# Sidebar content
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        1. Enter a valid **SMILES string** in the input box below.
        2. Click **Predict** to calculate the bioactivity.
        3. Download the prediction results as a CSV file.
        4. Example SMILES string: `CC(=O)OC1=CC=CC=C1C(=O)O`
        """
    )

# Main app title
st.title("Drug Bioactivity Prediction App")
st.write(
    """
    This app predicts the **bioactivity (pIC50)** of a drug based on its molecular structure.
    Enter a **SMILES string** below to get started.
    """
)

# User input for SMILES
smiles_input = st.text_area("Enter SMILES string(s):", height=100)
predict_button = st.button("Predict")

# If user clicks Predict
if predict_button:
    if not smiles_input.strip():
        st.warning("Please enter at least one valid SMILES string.")
    else:
        smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
        results = []

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol_weight = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)

                # Prediction
                input_features = [[mol_weight, logp]]
                prediction = model.predict(input_features)
                results.append({"SMILES": smiles, "MolWeight": mol_weight, "LogP": logp, "Predicted pIC50": prediction[0]})
            else:
                results.append({"SMILES": smiles, "MolWeight": None, "LogP": None, "Predicted pIC50": "Invalid SMILES"})

        # Create DataFrame from results
        results_df = pd.DataFrame(results)

        # Display results
        st.subheader("Prediction Results")
        st.write(results_df)

        # Provide download link
        st.markdown(create_download_link(results_df), unsafe_allow_html=True)


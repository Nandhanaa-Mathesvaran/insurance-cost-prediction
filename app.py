import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
WORK_DIR = Path.cwd()
MODELS_PATH = WORK_DIR / "insurance_models.pkl"
SCALER_PATH = WORK_DIR / "insurance_scaler.pkl"
COLUMNS_PATH = WORK_DIR / "insurance_columns.pkl"

CATEGORICAL_COLS = ['sex', 'smoker', 'region']
NUMERIC_COLS = ['age', 'bmi', 'children']

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    models = load_pickle(MODELS_PATH)
except Exception as e:
    st.error(f"Error loading models: {e}")
    models = None

try:
    scaler = load_pickle(SCALER_PATH)
except Exception as e:
    st.warning(f"Scaler not loaded: {e}")
    scaler = None

try:
    columns_list = load_pickle(COLUMNS_PATH)
except Exception as e:
    st.error(f"Columns file not found: {e}")
    columns_list = None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Insurance Cost Prediction", layout="centered")
st.title("💰 Insurance Cost Prediction")
st.write("Enter your details below to predict insurance charges:")

def user_input_features():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
    data = {
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region]
    }
    return pd.DataFrame(data)

input_df = user_input_features()

with st.expander("Input Features Preview"):
    st.write(input_df)

# -----------------------------
# Preprocess user input
# -----------------------------
# Log-transform children if it was log-transformed during training
input_df['children'] = np.log1p(input_df['children'])

# One-hot encode categorical features
input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, drop_first=True)

# Ensure all training columns exist
for col in columns_list:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns exactly as training
input_encoded = input_encoded[columns_list]

# Scale numerical features
if scaler is not None:
    input_scaled = scaler.transform(input_encoded)
else:
    input_scaled = input_encoded.values

# -----------------------------
# Predictions
# -----------------------------
if st.button("Predict Insurance Charges"):
    if models is None:
        st.error("Models not loaded.")
    else:
        results = {}
        for name, model in models.items():
            try:
                # Predict log-transformed charges
                pred_log = model.predict(input_scaled)
                # Convert back to original scale
                pred_value = np.expm1(pred_log)
                results[name] = float(pred_value.ravel()[0])
            except Exception as e:
                results[name] = f"Error: {e}"
        
        # Display predicted charges in original numeric values
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Predicted Charges'])
        st.table(results_df)
        st.success("Predictions generated successfully!")

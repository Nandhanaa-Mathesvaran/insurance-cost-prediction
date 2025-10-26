import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load saved objects ---
model = pickle.load(open("insurance_best_model.pkl", "rb"))
scaler = pickle.load(open("insurance_scaler.pkl", "rb"))
columns_list = pickle.load(open("insurance_columns.pkl", "rb"))

st.title("💰 Insurance Cost Prediction App")
st.write("Enter the details below to estimate medical charges.")

# --- User Inputs ---
age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker?", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Charges"):
    # --- Prepare input DataFrame ---
    input_dict = {
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex=='male' else 0],
        'smoker_yes': [1 if smoker=='yes' else 0],
        'region_northwest': [1 if region=='northwest' else 0],
        'region_southeast': [1 if region=='southeast' else 0],
        'region_southwest': [1 if region=='southwest' else 0]
    }

    # Ensure all columns exist
    for col in columns_list:
        if col not in input_dict:
            input_dict[col] = [0]

    input_df = pd.DataFrame(input_dict)[columns_list]

    # --- Scale only numerical columns ---
    num_cols = ['age', 'bmi', 'children']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # --- Predict ---
    prediction = model.predict(input_df)[0]

    st.success(f"✅ Predicted Insurance Cost: {prediction:,.2f}")

# app.py
import streamlit as st
import pandas as pd
import pickle

# Load scaler, columns, and models
scaler = pickle.load(open("insurance_scaler.pkl", "rb"))
columns = pickle.load(open("insurance_columns.pkl", "rb"))

model_names = ["GradientBoosting", "RandomForest", "XGBRegressor"]
models = {name: pickle.load(open(f"insurance_{name}.pkl", "rb")) for name in model_names}

# Hardcoded metrics for display
metrics = {
    "GradientBoosting": {"R2": 0.879, "MSE": 18792570, "RMSE": 4335},
    "RandomForest": {"R2": 0.864, "MSE": 21036680, "RMSE": 4587},
    "XGBRegressor": {"R2": 0.845, "MSE": 23992390, "RMSE": 4898},
}

# Streamlit UI
st.title("💰 Insurance Cost Prediction App")
st.write("Enter the details below to predict insurance charges:")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI Value", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Prepare input
input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
}

# Convert to DataFrame with correct column order
input_df = pd.DataFrame([input_dict])
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns]

# Scale numerical features
num_cols = ['age', 'bmi', 'children']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict for all models
if st.button("Predict"):
    st.subheader("Predictions for all models:")
    for name, model in models.items():
        pred = model.predict(input_df)[0]
        m = metrics[name]
        st.write(f"**{name}:**")
        st.write(f"- Predicted Insurance Cost: {round(pred, 2)}")
        st.write(f"- R² = {m['R2']}, MSE = {m['MSE']}, RMSE = {m['RMSE']}")
        st.write("---")

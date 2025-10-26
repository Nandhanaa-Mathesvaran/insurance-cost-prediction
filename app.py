import pandas as pd
import streamlit as st
import pickle

# ✅ Load models
with open("insurance_models.pkl", "rb") as f:
    trained_models = pickle.load(f)

st.title("Insurance Cost Prediction App")

age = st.number_input("Age", min_value=0, max_value=100, value=18)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=33.77)
children = st.number_input("Children", min_value=0, max_value=5, value=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

model_name = st.selectbox("Select Model", list(trained_models.keys()))

if st.button("Predict"):
    model = trained_models[model_name]

    # ✅ Must match training column names EXACTLY
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"💲 Predicted Medical Insurance Cost: ${prediction:.2f}")

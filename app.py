import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("mobile_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📱 Mobile Price Prediction App")

battery = st.number_input("Battery (mAh)", 1000, 8000, 4000)
ram = st.number_input("RAM (GB)", 1, 32, 4)
camera = st.number_input("Camera (MP)", 5, 200, 48)

if st.button("Predict Price"):
    input_data = np.array([[battery, ram, camera]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Price: {prediction[0]}")

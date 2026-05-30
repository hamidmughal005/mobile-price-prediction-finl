import streamlit as st
import pickle
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "mobile_price_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

st.title("📱 Mobile Price Prediction App")

ram = st.number_input("RAM")
battery = st.number_input("Battery")

if st.button("Predict"):
    data = np.array([[ram, battery]])
    prediction = model.predict(data)
    st.success(f"Predicted Price: {prediction[0]}")

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Food Delivery Time Prediction")

# Input fields
age = st.number_input("Age of Delivery Partner", min_value=18, max_value=60, value=30)
ratings = st.number_input("Ratings of Previous Deliveries", min_value=0.0, max_value=5.0, value=4.0)
distance = st.number_input("Total Distance (km)", min_value=0, max_value=50, value=5)

# Prediction
if st.button("Predict Delivery Time"):
    features = np.array([[age, ratings, distance]])
    prediction = model.predict(features)
    st.write(f"Predicted Delivery Time: {prediction[0]:.2f} minutes")

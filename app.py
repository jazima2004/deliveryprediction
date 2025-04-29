import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model
try:
    with open('model1.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model1.pkl' not found. Please ensure it is in the same directory as app.py.")
    st.stop()

# Streamlit app title
st.title("Delivery Time Prediction App")

# Description
st.write("Enter the details below to predict the delivery time (in minutes) using the pre-trained Random Forest model.")

# Create input fields for features
st.header("Input Features")

# Numerical inputs for the six features
delivery_person_age = st.number_input("Delivery Person Age", min_value=18.0, max_value=60.0, value=30.0, step=1.0)
delivery_person_ratings = st.number_input("Delivery Person Ratings", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
restaurant_latitude = st.number_input("Restaurant Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.0001, format="%.6f")
restaurant_longitude = st.number_input("Restaurant Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.0001, format="%.6f")
delivery_location_latitude = st.number_input("Delivery Location Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.0001, format="%.6f")
delivery_location_longitude = st.number_input("Delivery Location Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.0001, format="%.6f")

# Combine features into a single array
features = np.array([[
    delivery_person_age,
    delivery_person_ratings,
    restaurant_latitude,
    restaurant_longitude,
    delivery_location_latitude,
    delivery_location_longitude
]])

# Make prediction
if st.button("Predict Delivery Time"):
    try:
        prediction = model.predict(features)
        st.subheader("Prediction")
        st.write(f"Predicted Delivery Time: {prediction[0]:.2f} minutes")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Instructions for users
st.sidebar.header("Instructions")
st.sidebar.write("1. Enter the values for each feature (age, ratings, and location coordinates).")
st.sidebar.write("2. Click the 'Predict Delivery Time' button to see the prediction.")
st.sidebar.write("3. Ensure the model1.pkl file is in the same directory as this app.")

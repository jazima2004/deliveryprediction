import streamlit as st
from keras.models import load_model
import numpy as np

# Load the LSTM model
model = load_model('food_delivery_model.keras')

# Title of the app
st.title('Food Delivery Time Prediction 🚴‍♂️🍔')

# Input fields
age = st.number_input('Age of Delivery Partner', min_value=18, max_value=80, value=25)
rating = st.number_input('Ratings of Previous Deliveries', min_value=0.0, max_value=5.0, value=4.5, step=0.1)
distance = st.number_input('Total Distance (in KM)', min_value=0, max_value=100, value=5)

# Predict button
if st.button('Predict Delivery Time'):
    # Prepare input
    features = np.array([[age, rating, distance]])
    features = features.reshape((features.shape[0], features.shape[1], 1))  # Reshape for LSTM

    # Make prediction
    prediction = model.predict(features)
    predicted_time = prediction[0][0]

    # Show prediction
    st.success(f'Predicted Delivery Time: {predicted_time:.2f} minutes')

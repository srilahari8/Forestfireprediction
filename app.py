import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
with open('best_linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set page configuration for better look
st.set_page_config(
    page_title="Forest Fires Area Prediction",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# App Title and Description
st.title('🔥 Predict Burned Forest Area Based on Weather Parameters')

st.write("""
    This app predicts the burned area of the forest based on weather conditions:
    - *Temperature (Celsius)*: Temperature in Celsius.
    - *Relative Humidity (%)*: Humidity as a percentage.
    - *Wind Speed (km/h)*: Wind speed.
    - *Rain (mm/m²)*: Rainfall in millimeters per square meter.
""")

# Organizing layout into two columns
col1, col2 = st.columns(2)

with col1:
    # Sliders for numerical input
    temp = st.slider('Temperature (Celsius)', min_value=0.0, max_value=50.0, value=15.0)
    RH = st.slider('Relative Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)

with col2:
    wind = st.slider('Wind Speed (km/h)', min_value=0.0, max_value=20.0, value=5.0)
    rain = st.slider('Rain (mm/m²)', min_value=0.0, max_value=10.0, step=0.1, value=0.0)

# Predict button
if st.button('Predict Burned Area'):
    # Prepare input data
    input_data = pd.DataFrame([[temp, RH, wind, rain]], 
                              columns=['temp', 'RH', 'wind', 'rain'])

    # Ensure the input data columns match the scaler's expected columns
    expected_columns = scaler.feature_names_in_  # Extracts feature names used when the scaler was fit
    input_data = input_data[expected_columns]    # Reorders or selects columns to match

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Display prediction
    st.write(f'🔥 *Predicted Burned Area (in hectares)*: {prediction[0]:.2f} hectares')

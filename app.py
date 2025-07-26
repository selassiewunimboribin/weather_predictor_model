import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Define the label mapping
label_map = {
    0: 'Clear',
    1: 'Cloudy',
    2: 'Fog',
    3: 'Other',
    4: 'Rain'
}

# App title
st.title("🌦️ Weather Classifier")

st.markdown("Enter the weather parameters below to predict the weather condition.")

# Input fields with styled labels and unique keys
st.markdown('<p style="font-weight:bold; color:black;">Temperature (°C)</p>', unsafe_allow_html=True)
temp = st.number_input("", value=20.0, key="temp_input")

st.markdown('<p style="font-weight:bold; color:black;">Dew Point Temperature (°C)</p>', unsafe_allow_html=True)
dew_point = st.number_input("", value=10.0, key="dew_input")

st.markdown('<p style="font-weight:bold; color:black;">Relative Humidity (%)</p>', unsafe_allow_html=True)
rel_hum = st.number_input("", value=50.0, key="hum_input")

st.markdown('<p style="font-weight:bold; color:black;">Wind Speed (km/h)</p>', unsafe_allow_html=True)
wind_speed = st.number_input("", value=10.0, key="wind_input")

st.markdown('<p style="font-weight:bold; color:black;">Visibility (km)</p>', unsafe_allow_html=True)
visibility = st.number_input("", value=20.0, key="vis_input")

st.markdown('<p style="font-weight:bold; color:black;">Pressure (kPa)</p>', unsafe_allow_html=True)
press = st.number_input("", value=101.0, key="press_input")

st.markdown('<p style="font-weight:bold; color:black;">Day of the Month</p>', unsafe_allow_html=True)
day = st.slider("", 1, 31, 15, key="day_input")

st.markdown('<p style="font-weight:bold; color:black;">Month</p>', unsafe_allow_html=True)
month = st.slider("", 1, 12, 6, key="month_input")

# Feature engineering (must match model training)
temp_spread = temp - dew_point
humidity_ratio = rel_hum / 100.0
wind_chill = temp - (0.7 * wind_speed)
press_diff = press - 100
temp_wind_interaction = temp * wind_speed
press_vis_ratio = press / (visibility + 0.1)  # Avoid division by zero
visibility_risk = 1 if visibility < 5 else 0

# Arrange inputs for model prediction
features = np.array([[
    temp, dew_point, rel_hum, wind_speed, visibility, press,
    visibility_risk, temp_spread, humidity_ratio, wind_chill,
    press_diff, temp_wind_interaction, press_vis_ratio
]])

# Prediction button
if st.button("Predict Weather"):
    prediction = model.predict(features)
    predicted_label = label_map[int(prediction[0])]

    st.markdown(
        f"<h3 style='color:black;'>Predicted Weather: <span style='font-weight:bold;'>{predicted_label}</span></h3>",
        unsafe_allow_html=True
    )

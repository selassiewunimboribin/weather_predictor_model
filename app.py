import streamlit as st
import pandas as pd
import joblib
import base64

# Load model
model = joblib.load("model.pkl")

st.set_page_config(layout="wide")

# Function to map month to season
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

# Function to set background image from file
def set_bg_from_image(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- UI Layout ---
st.title("🌦️ Weather Classification App")
st.markdown("Enter weather conditions to predict the class (e.g., Clear, Cloudy, Rain, etc.)")

# --- Input columns ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<span style='font-weight:bold; color:black;'>Temperature (°C)</span>", unsafe_allow_html=True)
    temp = st.number_input("", value=20.0)

    st.markdown("<span style='font-weight:bold; color:black;'>Visibility (km)</span>", unsafe_allow_html=True)
    visibility = st.number_input("", value=20.0)

with col2:
    st.markdown("<span style='font-weight:bold; color:black;'>Relative Humidity (%)</span>", unsafe_allow_html=True)
    rel_hum = st.number_input("", value=50.0)

    st.markdown("<span style='font-weight:bold; color:black;'>Pressure (kPa)</span>", unsafe_allow_html=True)
    press = st.number_input("", value=101.0)

with col3:
    st.markdown("<span style='font-weight:bold; color:black;'>Wind Speed (km/h)</span>", unsafe_allow_html=True)
    wind_speed = st.number_input("", value=10.0)

    st.markdown("<span style='font-weight:bold; color:black;'>Day of the Month</span>", unsafe_allow_html=True)
    day = st.slider("", 1, 31, 15)

# Month slider and season
st.markdown("<br>", unsafe_allow_html=True)
month = st.slider("Month", 1, 12, 6)
season = get_season(month)
st.info(f"Season for Month {month}: **{season}**")

# Set background based on season
background_path = f"images/{season.lower()}.jpg"
set_bg_from_image(background_path)

# One-hot encode season
season_cols = {
    "season_Fall": 0,
    "season_Spring": 0,
    "season_Summer": 0,
    "season_Winter": 0
}
season_cols[f"season_{season}"] = 1

# Predict button
if st.button("Predict Weather"):
    input_dict = {
        "Temp_C": temp,
        "Rel Hum_%": rel_hum,
        "Wind Speed_km/h": wind_speed,
        "Visibility_km": visibility,
        "Press_kPa": press,
        "Day": day,
        "Month": month,
        **season_cols
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_df)[0]

    label_mapping = {
        0: "Clear",
        1: "Cloudy",
        2: "Drizzle",
        3: "Fog",
        4: "Other",
        5: "Rain",
        6: "Snow"
    }
    predicted_label = label_mapping[prediction]

    # Display styled result
    st.markdown(
        f"""
        <div style="padding: 1em; background-color: rgba(255, 255, 255, 0.75); 
                    border-radius: 10px; display: inline-block; 
                    border: 2px solid black;">
            <h3 style="color: black;">Predicted Weather: {predicted_label}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

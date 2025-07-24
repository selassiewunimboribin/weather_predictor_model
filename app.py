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

# --- User inputs ---
st.title("🌦️ Weather Classification App")
st.markdown("Enter weather conditions to predict the class (e.g., Clear, Cloudy, Rain, etc.)")

temp = st.number_input("Temperature (°C)", value=20.0)
rel_hum = st.number_input("Relative Humidity (%)", value=50.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
visibility = st.number_input("Visibility (km)", value=20.0)
press = st.number_input("Pressure (kPa)", value=101.0)
day = st.slider("Day of the Month", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)

# --- Determine season ---
season = get_season(month)
st.info(f"Season for Month {month}: **{season}**")

# --- Set background image ---
background_path = f"images/{season.lower()}.jpg"
set_bg_from_image(background_path)

# --- One-hot encoding for season ---
season_cols = {
    "season_Fall": 0,
    "season_Spring": 0,
    "season_Summer": 0,
    "season_Winter": 0
}
season_cols[f"season_{season}"] = 1

# --- Predict button ---
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

    # --- Styled result display ---
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

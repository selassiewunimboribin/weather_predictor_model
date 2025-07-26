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

st.markdown("<strong style='color:black;'>Temperature (°C)</strong>", unsafe_allow_html=True)
temp = st.number_input("", value=20.0, key="temp")

st.markdown("<strong style='color:black;'>Relative Humidity (%)</strong>", unsafe_allow_html=True)
rel_hum = st.number_input("", value=50.0, key="humidity")

st.markdown("<strong style='color:black;'>Wind Speed (km/h)</strong>", unsafe_allow_html=True)
wind_speed = st.number_input("", value=10.0, key="wind")

st.markdown("<strong style='color:black;'>Visibility (km)</strong>", unsafe_allow_html=True)
visibility = st.number_input("", value=20.0, key="visibility")

st.markdown("<strong style='color:black;'>Pressure (kPa)</strong>", unsafe_allow_html=True)
press = st.number_input("", value=101.0, key="pressure")

st.markdown("<strong style='color:black;'>Day of the Month</strong>", unsafe_allow_html=True)
day = st.slider("", 1, 31, 15, key="day")

st.markdown("<strong style='color:black;'>Month</strong>", unsafe_allow_html=True)
month = st.slider("", 1, 12, 6, key="month")


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
        "Press_KPa": press,
        "Day": day,
        "Month": month,
        **season_cols
    }

    input_df = pd.DataFrame([input_dict])

    # Make sure the column order matches what the model expects
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # --- Debugging outputs ---
    st.subheader(" Debug Info")
    st.write("Model expects these features:", list(model.feature_names_in_))
    st.write(" DataFrame being passed to model:")
    st.dataframe(input_df)

    # --- Prediction ---
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
    predicted_label = label_mapping.get(prediction, "Unknown")

    # Display result
    st.markdown(
        f"""
        <div style="padding: 1em; background-color: rgba(255, 255, 255, 0.7); border-radius: 10px; display: inline-block;">
            <h3 style="color: black;">Predicted Weather: {predicted_label}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

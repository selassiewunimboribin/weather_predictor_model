import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------
# Load and preprocess data
# ----------------------
weather_data = pd.read_csv(r"C:\Users\DELL\Downloads\datasets\Weather Data.csv") 
weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])

# Extract date parts
weather_data['Day'] = weather_data['Date/Time'].dt.day
weather_data['Month'] = weather_data['Date/Time'].dt.month
weather_data['Year'] = weather_data['Date/Time'].dt.year

# Drop unnecessary columns
weather_data = weather_data.drop(columns=['Date/Time', 'Year', 'Dew Point Temp_C'])
weather_data = weather_data.sample(frac=1).reset_index().drop(columns=['index'])

# Simplify weather conditions
mapping = {
    'Sunny': 'Clear', 'Clear': 'Clear', 'Mostly Sunny': 'Clear', 'Mainly Clear': 'Clear',
    'Cloudy': 'Cloudy', 'Partly Cloudy': 'Cloudy', 'Mostly Cloudy': 'Cloudy',
    'Rain': 'Rain', 'Heavy Rain': 'Rain', 'Light Rain': 'Rain', 'Freezing Rain': 'Rain',
    'Rain,Fog': 'Rain', 'Rain Showers': 'Rain', 'Thunderstorms,Heavy Rain Showers': 'Rain',
    'Thunderstorms,Rain Showers': 'Rain', 'Thunderstorms,Rain': 'Rain',
    'Thunderstorms,Rain Showers,Fog': 'Rain', 'Rain,Snow': 'Rain', 'Rain,Haze': 'Rain',
    'Freezing Rain,Fog': 'Rain', 'Freezing Rain,Haze': 'Rain', 'Rain,Snow,Ice Pellets': 'Rain',
    'Freezing Rain,Ice Pellets,Fog': 'Rain', 'Freezing Rain,Snow Grains': 'Rain',
    'Moderate Rain,Fog': 'Rain', 'Rain Showers,Fog': 'Rain', 'Rain Showers,Snow Showers': 'Rain',
    'Rain,Ice Pellets': 'Rain', 'Rain,Snow Grains': 'Rain', 'Rain,Snow,Fog': 'Rain',
    'Thunderstorms': 'Rain', 'Thunderstorms,Moderate Rain Showers,Fog': 'Rain',
    'Thunderstorms,Rain,Fog': 'Rain',
    'Drizzle': 'Drizzle', 'Drizzle,Fog': 'Drizzle', 'Drizzle,Ice Pellets,Fog': 'Drizzle',
    'Freezing Drizzle': 'Drizzle', 'Freezing Drizzle,Snow': 'Drizzle', 'Freezing Drizzle,Fog': 'Drizzle',
    'Freezing Drizzle,Haze': 'Drizzle', 'Drizzle,Snow': 'Drizzle', 'Drizzle,Snow,Fog': 'Drizzle',
    'Snow': 'Snow', 'Snow Showers': 'Snow', 'Snow,Fog': 'Snow', 'Snow,Haze': 'Snow',
    'Snow,Blowing Snow': 'Snow', 'Snow,Ice Pellets': 'Snow', 'Snow Showers,Fog': 'Snow',
    'Moderate Snow': 'Snow', 'Moderate Snow,Blowing Snow': 'Snow', 'Snow Pellets': 'Snow',
    'Fog': 'Fog', 'Mist': 'Fog', 'Haze': 'Fog', 'Freezing Fog': 'Fog'
}

weather_data['Weather_condition'] = weather_data['Weather'].map(mapping).fillna('Other')

# Define a function to map months to seasons
def month_to_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

weather_data['Season'] = weather_data['Month'].apply(month_to_season)

# Drop original weather description
weather_data = weather_data.drop(columns=['Weather'])
weather_data = weather_data.round(2)

# Label encode target
le_simple = LabelEncoder()
weather_data['Weather_condition'] = le_simple.fit_transform(weather_data['Weather_condition'])

# One-hot encode Season
weather_data = pd.get_dummies(weather_data, columns=['Season'], drop_first=True)

# ----------------------
# Split Data
# ----------------------
independent_values = weather_data.drop("Weather_condition", axis=1)
dependent_values = weather_data["Weather_condition"]

ind_train, ind_temp, dep_train, dep_temp = train_test_split(
    independent_values, dependent_values,
    test_size=0.30, random_state=42, stratify=dependent_values
)
ind_valid, ind_test, dep_valid, dep_test = train_test_split(
    ind_temp, dep_temp,
    test_size=0.50, random_state=42, stratify=dep_temp
)

# ----------------------
# Train Model
# ----------------------
rfc = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    min_samples_split=2,
    max_depth=40,
    criterion='log_loss',
    max_features='log2'
)
rfc.fit(ind_train, dep_train)
dep_pred_rfc = rfc.predict(ind_valid)

# ----------------------
# Evaluation
# ----------------------
def evaluation_matrices(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    print("\n Classification Report:\n", classification_report(y_true, y_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluation_matrices(dep_valid, dep_pred_rfc)

import joblib

# Save the trained model
joblib.dump(rfc, "model.pkl", compress=3)
print("Model saved successfully as model.pkl")

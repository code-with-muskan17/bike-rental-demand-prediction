import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="centered")

# Load saved model, scaler arrays and column order (cached so it loads only once)
@st.cache_resource
def load_artifacts():
    model = XGBRegressor()
    model.load_model("best_bike_rental_model.json")
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")
    columns = joblib.load("model_columns.pkl")
    return model, scaler_mean, scaler_scale, columns

model, scaler_mean, scaler_scale, model_columns = load_artifacts()

st.title("🚲 Bike Rental Demand Predictor")
st.write("Enter the conditions below to predict the expected number of hourly bike rentals.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        yr_label = st.selectbox("Year", options=[2011, 2012], index=1)
        yr = 0 if yr_label == 2011 else 1
        mnth = st.slider("Month", 1, 12, 6)
        hr = st.slider("Hour of Day", 0, 23, 17)
        weekday = st.selectbox(
            "Day of Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday",
                                    "Thursday", "Friday", "Saturday"][x],
            index=3
        )
        season = st.selectbox("Season", options=["springer", "summer", "fall", "winter"], index=1)

    with col2:
        temp = st.slider("Temperature (normalized 0-1)", 0.0, 1.0, 0.6, 0.01)
        hum = st.slider("Humidity (normalized 0-1)", 0.0, 1.0, 0.5, 0.01)
        windspeed = st.slider("Windspeed (normalized 0-1)", 0.0, 1.0, 0.2, 0.01)
        holiday = st.selectbox("Holiday", options=["No", "Yes"])
        workingday = st.selectbox("Working Day", options=["Working Day", "No work"])
        weathersit = st.selectbox("Weather Situation", options=["Clear", "Mist", "Light Snow", "Heavy Rain"])

    submitted = st.form_submit_button("Predict Rentals")


def predict_rentals(input_dict, model, scaler_mean, scaler_scale, columns):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df[["temp", "hum", "windspeed"]] = (df[["temp", "hum", "windspeed"]] - scaler_mean) / scaler_scale
    prediction = model.predict(df)
    return max(0, round(prediction[0]))


if submitted:
    is_weekend = 1 if weekday in [0, 6] else 0
    quarter = (mnth - 1) // 3 + 1
    is_summer = 1 if mnth in [5, 6, 7, 8] else 0
    is_winter = 1 if mnth in [11, 12, 1, 2] else 0

    input_dict = {
        "yr": yr, "mnth": mnth, "hr": hr, "weekday": weekday,
        "temp": temp, "hum": hum, "windspeed": windspeed,
        "is_weekend": is_weekend, "quarter": quarter,
        "is_summer": is_summer, "is_winter": is_winter,
        "season": season, "holiday": holiday,
        "workingday": workingday, "weathersit": weathersit
    }

    predicted = predict_rentals(input_dict, model, scaler_mean, scaler_scale, model_columns)

    st.success(f"### Predicted Bike Rentals: **{predicted}**")
    st.caption("Model: XGBoost | R² = 0.950 | MAE ≈ 24.5 rentals")

with st.sidebar:
    st.header("Model Info")
    st.write("**Algorithm:** XGBoost Regressor")
    st.write("**R² Score:** 0.950")
    st.write("**MAE:** 24.51")
    st.write("**RMSE:** 39.96")
    st.write("Trained on the UCI Bike Sharing Dataset (2011-2012).")

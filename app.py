import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="wide")

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


def predict_rentals(input_dict, model, scaler_mean, scaler_scale, columns):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df[["temp", "hum", "windspeed"]] = (df[["temp", "hum", "windspeed"]] - scaler_mean) / scaler_scale
    prediction = model.predict(df)
    return max(0, round(prediction[0]))


# Sidebar inputs
with st.sidebar:
    st.markdown("### 🚲 Set Conditions")
    st.divider()

    st.markdown("#### ⏰ Time Features")
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
    st.divider()

    st.markdown("#### 🌤 Weather & Conditions")
    season = st.selectbox("Season", options=["springer", "summer", "fall", "winter"], index=1)
    weathersit = st.selectbox("Weather", options=["Clear", "Mist", "Light Snow", "Heavy Rain"])
    temp = st.slider("Temperature (0-1)", 0.0, 1.0, 0.6, 0.01)
    hum = st.slider("Humidity (0-1)", 0.0, 1.0, 0.5, 0.01)
    windspeed = st.slider("Windspeed (0-1)", 0.0, 1.0, 0.2, 0.01)
    st.divider()

    st.markdown("#### 📅 Day Type")
    holiday = st.selectbox("Holiday", options=["No", "Yes"])
    workingday = st.selectbox("Working Day", options=["Working Day", "No work"])
    st.divider()

    predict_btn = st.button("🔮 Predict Rentals", use_container_width=True, type="primary")
    st.divider()

    st.caption("Model: XGBoost | R²=0.950")
    st.caption("MAE=24.51 | RMSE=39.96")
    st.caption("UCI Bike Sharing Dataset (2011-2012)")


# Main area
st.markdown("# 🚲 Bike Rental Demand Predictor")
st.write("Predict the expected number of hourly bike rentals based on weather and time conditions.")
st.divider()

# Prediction result
if predict_btn:
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

    # Demand level based on actual data distribution (Q1=40, Q3=281)
    if predicted <= 40:
        demand_color = "#c0392b"
        demand_label = "Low Demand"
        demand_icon = "🔴"
    elif predicted <= 280:
        demand_color = "#d4a017"
        demand_label = "Medium Demand"
        demand_icon = "🟡"
    else:
        demand_color = "#27ae60"
        demand_label = "High Demand"
        demand_icon = "🟢"

    # Prediction card — clean HTML, no anchor icon
    st.markdown(f"""
    <div style="background-color:{demand_color};padding:40px;border-radius:12px;text-align:center;">
        <p style="color:white;font-size:14px;letter-spacing:2px;margin:0;">🎯 PREDICTED HOURLY RENTALS</p>
        <p style="color:white;font-size:80px;font-weight:bold;margin:10px 0;">{predicted}</p>
        <p style="color:white;font-size:18px;margin:0;">bikes expected this hour</p>
        <br>
        <p style="color:white;font-size:16px;margin:0;">{demand_icon} {demand_label}</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Before prediction state
    st.markdown("""
    <div style="background-color:#1e1e2e;padding:40px;border-radius:12px;text-align:center;border:2px dashed #555;">
        <p style="color:#aaa;font-size:18px;margin:0;">👈 Set your conditions in the sidebar and click <strong>Predict Rentals</strong></p>
    </div>
    """, unsafe_allow_html=True)

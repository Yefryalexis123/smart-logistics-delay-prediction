import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "logistics_delay_model.pkl"
)

# Cargar modelo
model = joblib.load("models/logistics_delay_model.pkl")
st.set_page_config(page_title="Logistics Delay Predictor")

st.title("🚚 Logistics Delay Prediction")

st.write(
    "Predice si un envío tendrá retraso utilizando variables operativas y ambientales."
)

# Entradas

latitude = st.number_input("Latitude", value=40.0)

longitude = st.number_input("Longitude", value=-74.0)

inventory_level = st.slider(
    "Inventory Level",
    min_value=0,
    max_value=1000,
    value=500
)

temperature = st.slider(
    "Temperature",
    min_value=-10.0,
    max_value=50.0,
    value=25.0
)

humidity = st.slider(
    "Humidity",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

traffic_status = st.selectbox(
    "Traffic Status",
    ["Clear", "Heavy", "Detour"]
)

waiting_time = st.slider(
    "Waiting Time",
    min_value=0,
    max_value=120,
    value=30
)

user_transaction_amount = st.number_input(
    "User Transaction Amount",
    value=100
)

user_purchase_frequency = st.slider(
    "User Purchase Frequency",
    min_value=0,
    max_value=20,
    value=5
)

asset_utilization = st.slider(
    "Asset Utilization",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

demand_forecast = st.slider(
    "Demand Forecast",
    min_value=0,
    max_value=1000,
    value=500
)

if st.button("Predict Delay"):

    data = pd.DataFrame([{
        "Latitude": latitude,
        "Longitude": longitude,
        "Inventory_Level": inventory_level,
        "Temperature": temperature,
        "Humidity": humidity,
        "Traffic_Status": traffic_status,
        "Waiting_Time": waiting_time,
        "User_Transaction_Amount": user_transaction_amount,
        "User_Purchase_Frequency": user_purchase_frequency,
        "Asset_Utilization": asset_utilization,
        "Demand_Forecast": demand_forecast
    }])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]

    if prediction == 1:
        st.error(
            f"⚠️ Delay predicted ({probability[1]:.2%} confidence)"
        )
    else:
        st.success(
            f"✅ No delay predicted ({probability[0]:.2%} confidence)"
        )
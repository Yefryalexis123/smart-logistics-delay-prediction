from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

from src.predict import load_model

model = load_model(
    BASE_DIR / "models" / "logistics_delay_model.pkl"
)

st.set_page_config(
    page_title="Predicción de Retrasos Logísticos"
)


st.markdown("""
Esta aplicación utiliza un modelo Random Forest para predecir
posibles retrasos en envíos logísticos utilizando variables
operativas, ambientales y de demanda.
""")

# Cargar modelo
model = joblib.load("models/logistics_delay_model.pkl")
st.set_page_config(page_title="Logistics Delay Predictor")

st.title("🚚  Predicción de Retrasos Logísticos")

st.write(
    "Predice si un envío tendrá retraso utilizando variables operativas y ambientales."
)

# Entradas

latitude = st.number_input("Latitud", value=40.0)

longitude = st.number_input("Longitud", value=-74.0)

inventory_level = st.slider(
    "Nivel de Inventario",
    min_value=0,
    max_value=1000,
    value=500
)

temperature = st.slider(
    "Temperatura",
    min_value=-10.0,
    max_value=50.0,
    value=25.0
)

humidity = st.slider(
    "Humedad",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

traffic_status = st.selectbox(
    "Estado del Tráfico",
    ["Clear", "Heavy", "Detour"]
)

waiting_time = st.slider(
    "Tiempo de Espera",
    min_value=0,
    max_value=120,
    value=30
)

user_transaction_amount = st.number_input(
    "Monto de Transacción",
    value=100
)

user_purchase_frequency = st.slider(
    "Frecuencia de Compras del Usuario",
    min_value=0,
    max_value=20,
    value=5
)

asset_utilization = st.slider(
    "Utilización de Recursos (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

demand_forecast = st.slider(
    "Demanda Estimada",
    min_value=0,
    max_value=1000,
    value=500
)

if st.button("Realizar Predicción"):

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

    st.subheader("Resultado de la Predicción")

    st.write(
        f"Probabilidad de retraso: {probability[1] * 100:.2f}%"
    )

    st.write(
        f"Probabilidad de entrega sin retraso: {probability[0] * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ Se predice un posible retraso")
    else:
        st.success("✅ No se prevé retraso")
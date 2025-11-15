import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load Model and Scalers
model = load_model("ann_crop_model.h5", compile=False)   # <-- FIXED
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

df = pd.read_csv("preprocessed_crop_yield_data.csv")


# -------------------------------------------
# Build feature vector
# -------------------------------------------
def make_feature_vector(state, crop, rainfall, temp, humidity, N, P, K, ph, year):

    row = {
        'year': year,
        'rainfall_mm': rainfall,
        'temperature_C': temp,
        'humidity_pct': humidity,
        'soil_N_kg_ha': N,
        'soil_P_kg_ha': P,
        'soil_K_kg_ha': K,
        'soil_pH': ph,
    }

    # Add encoded state columns
    for col in df.columns:
        if col.startswith("state_"):
            row[col] = 1 if col == f"state_{state}" else 0

    # Add encoded crop columns
    for col in df.columns:
        if col.startswith("crop_"):
            row[col] = 1 if col == f"crop_{crop}" else 0

    # Return as DataFrame ready for scaling
    return pd.DataFrame([row])

# -------------------------------------------
# ANN Prediction Function
# -------------------------------------------
def predict_yield(input_df):
    x_scaled = x_scaler.transform(input_df)
    pred_scaled = model.predict(x_scaled)

    # Ensure correct shape for inverse transform
    pred_scaled = np.array(pred_scaled).reshape(-1, 1)

    pred_real = y_scaler.inverse_transform(pred_scaled)[0][0]
    return float(pred_real)

# -------------------------------------------
# Fertilizer Recommendation (Simple GA)
# -------------------------------------------
def recommend_fertilizer(base_features):
    import random
    best_yield = -1
    best_npk = (0, 0, 0)

    for _ in range(50):
        N = random.uniform(40, 150)
        P = random.uniform(10, 80)
        K = random.uniform(20, 120)

        base_features['soil_N_kg_ha'] = N
        base_features['soil_P_kg_ha'] = P
        base_features['soil_K_kg_ha'] = K

        current_yield = predict_yield(base_features)

        if current_yield > best_yield:
            best_yield = current_yield
            best_npk = (N, P, K)

    return best_npk, best_yield

# -------------------------------------------
# Streamlit UI
# -------------------------------------------
st.title("🌾 Intelligent Crop Yield & Fertilizer Advisor")
st.write("AI-based yield prediction and fertilizer optimization")

# State dropdown
state = st.selectbox(
    "Select State",
    [s.replace("state_", "") for s in df.columns if s.startswith("state_")]
)

# Crop dropdown
crop = st.selectbox(
    "Select Crop",
    [s.replace("crop_", "") for s in df.columns if s.startswith("crop_")]
)

# Input Fields
year = st.slider("Year", 2020, 2025, 2024)
rainfall = st.number_input("Rainfall (mm)", 0.0, 3000.0, 900.0)
temp = st.number_input("Temperature (°C)", 5.0, 45.0, 28.0)
humidity = st.number_input("Humidity (%)", 10.0, 100.0, 65.0)
ph = st.number_input("Soil pH", 3.0, 10.0, 6.5)

N = st.number_input("Current Nitrogen (kg/ha)", 0.0, 300.0, 40.0)
P = st.number_input("Current Phosphorus (kg/ha)", 0.0, 200.0, 20.0)
K = st.number_input("Current Potassium (kg/ha)", 0.0, 250.0, 35.0)

# Build Input Feature Vector
feature_df = make_feature_vector(state, crop, rainfall, temp, humidity, N, P, K, ph, year)

# -------------------------------
# Buttons
# -------------------------------
if st.button("Predict Yield"):
    pred = predict_yield(feature_df)
    st.success(f"🌾 Predicted Yield: **{pred:.2f} kg/ha**")

if st.button("Recommend Optimal Fertilizer"):
    best_npk, best_yield = recommend_fertilizer(feature_df.copy())
    st.success("🌿 Recommended Fertilizer (kg/ha):")
    st.write(f"➡ Nitrogen (N): **{best_npk[0]:.2f}**")
    st.write(f"➡ Phosphorus (P): **{best_npk[1]:.2f}**")
    st.write(f"➡ Potassium (K): **{best_npk[2]:.2f}**")
    st.info(f"📈 Expected Maximum Yield: **{best_yield:.2f} kg/ha**")

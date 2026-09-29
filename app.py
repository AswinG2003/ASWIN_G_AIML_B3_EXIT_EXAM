
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# -----------------------------
# Load trained model and scaler
# -----------------------------
model = load_model("asteroid_dnn.keras")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Asteroid Diameter Predictor",
    page_icon="☄️",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("☄️ Asteroid Diameter Predictor")

st.write(
    "Enter the physical and orbital characteristics of an asteroid "
    "to predict its diameter."
)


# -----------------------------
# User inputs
# -----------------------------
st.header("Asteroid Information")

H = st.number_input(
    "Absolute Magnitude (H)",
    value=15.0
)

albedo = st.number_input(
    "Albedo",
    min_value=0.0,
    value=0.10
)

e = st.number_input(
    "Eccentricity (e)",
    min_value=0.0,
    value=0.10
)

a = st.number_input(
    "Semi-major Axis (a)",
    min_value=0.0,
    value=2.50
)

q = st.number_input(
    "Perihelion Distance (q)",
    min_value=0.0,
    value=2.00
)

i = st.number_input(
    "Inclination (i)",
    min_value=0.0,
    value=5.0
)

n = st.number_input(
    "Mean Motion (n)",
    min_value=0.0,
    value=0.20
)

moid = st.number_input(
    "MOID",
    min_value=0.0,
    value=1.0
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Diameter"):

    # Arrange inputs in the same order
    # used during model training
    input_data = np.array([
        [H, albedo, e, a, q, i, n, moid]
    ])

    # Apply the SAME scaler used during training
    input_scaled = scaler.transform(input_data)

    # Predict log-transformed diameter
    prediction_log = model.predict(
        input_scaled,
        verbose=0
    )

    # Convert log prediction back to actual diameter
    prediction = np.expm1(prediction_log)[0][0]

    st.success(
        f"☄️ Predicted Asteroid Diameter: {prediction:.2f} km"
    )

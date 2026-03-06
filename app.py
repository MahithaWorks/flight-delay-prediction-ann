import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Page configuration
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈",
    layout="wide"
)

# Load trained model
model = tf.keras.models.load_model("flight_delay_model.h5")

# Load scaler
with open("scaler.pkl", "rb") as f:
    sc = pickle.load(f)

# Title
st.title("✈ Flight Delay Prediction Dashboard")

st.markdown(
"""
This system predicts whether a flight will **arrive on time or be delayed**
using a trained **Artificial Neural Network (ANN)** model.
"""
)

st.divider()

# Sidebar Inputs
st.sidebar.header("Enter Flight Details")

id_val = st.sidebar.number_input("Flight ID", min_value=0)
airline = st.sidebar.number_input("Airline Code", min_value=0)
flight = st.sidebar.number_input("Flight Number", min_value=0)
airport_from = st.sidebar.number_input("Departure Airport Code", min_value=0)
airport_to = st.sidebar.number_input("Arrival Airport Code", min_value=0)
day = st.sidebar.number_input("Day of Week (1-7)", min_value=1, max_value=7)
time = st.sidebar.number_input("Departure Time", min_value=0)
length = st.sidebar.number_input("Flight Length (minutes)", min_value=0)

st.divider()

# Prediction section
st.subheader("Prediction")

if st.button("🔍 Predict Flight Status"):

    # Prepare input data
    input_data = np.array([[id_val, airline, flight,
                            airport_from, airport_to,
                            day, time, length]])

    # Scale input
    input_data = sc.transform(input_data)

    # Model prediction
    prediction = model.predict(input_data)

    probability = float(prediction)

    st.write("Prediction Probability:", round(probability, 3))

    # Progress bar
    st.progress(probability)

    # Threshold for delay
    threshold = 0.45

    if probability > threshold:

        st.error("⚠ Flight is likely to be DELAYED")

        # Estimated delay time
        delay_time = int(probability * 120)

        st.write("Estimated Delay Time:", delay_time, "minutes")

    else:
        st.success("✅ Flight is likely to be ON TIME")

st.divider()

# Project information dashboard
st.subheader("Project Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "539K Flights")

with col2:
    st.metric("Model Accuracy", "66%")

with col3:
    st.metric("Model Type", "Artificial Neural Network")

st.caption("Developed using Python, TensorFlow, and Streamlit.")
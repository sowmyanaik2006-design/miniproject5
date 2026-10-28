import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load trained model
MODEL_FILE = "house_model.pkl"

st.title("🏠 House Price Prediction")
st.write("Predict house prices using Linear Regression.")

if not os.path.exists(MODEL_FILE):
    st.warning("Model not found. Please train the model first.")
    st.code("python train_model.py")
    st.stop()

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

st.subheader("Enter House Details")

# User inputs
area = st.number_input(
    "Area (square feet)",
    min_value=100,
    max_value=10000,
    value=1500,
    step=100
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

age = st.number_input(
    "Age of House (years)",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

# Prediction button
if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame(
        [[area, bedrooms, age]],
        columns=["Area", "Bedrooms", "Age"]
    )

    predicted_price = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        "Estimated House Price",
        f"₹ {predicted_price:.2f} Lakhs"
    )

    st.caption(
        "This estimate uses sample training data and is not "
        "a real market valuation."
    )

st.divider()
st.caption("Project: House Price Prediction using Linear Regression")
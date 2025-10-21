import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pickled objects
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

st.title("Parkinson's Disease Prediction App")

st.write("Input values for each feature:")

def user_input_features():
    # Create input widgets for all features in the right order
    input_dict = {}
    for col in columns:
        input_dict[col] = st.number_input(col, value=0.0)
    return pd.DataFrame([input_dict])

# Get user inputs, scale them, and predict
input_df = user_input_features()

if st.button('Predict Parkinson\'s Status'):
    # It's important the input columns match model training
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0, 1] if hasattr(model, 'predict_proba') else None

    st.write(f"Prediction: {'Parkinson\'s Disease' if prediction == 1 else 'Healthy'}")
    if proba is not None:
        st.write(f"Probability of Parkinson's Disease: {proba:.2f}")

st.write("Model: Loaded from pickle file")

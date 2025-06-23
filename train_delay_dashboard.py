# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:57:33 2025

@author: kavin
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

st.set_page_config(page_title="Train Delay Dashboard", layout="wide")

cris_logo = Image.open("logo_cris.png")
st.image(cris_logo, width=150)
st.title("Train Delay Analysis & Prediction Dashboard")

@st.cache_data
def load_data():
    info = pd.read_csv("train_info.csv")
    schedule = pd.read_csv("train_schedule.csv")
    delay = pd.read_csv("train delay data.csv")
    return info, schedule, delay

info_df, schedule_df, delay_df = load_data()

def clean_data():
    schedule_df['Arrival_time'] = pd.to_datetime(schedule_df['Arrival_time'], errors='coerce').dt.time
    schedule_df['Departure_Time'] = pd.to_datetime(schedule_df['Departure_Time'], errors='coerce').dt.time
    delay_df['Historical Delay (min)'] = pd.to_numeric(delay_df['Historical Delay (min)'], errors='coerce')
    delay_df['On_Time'] = delay_df['Historical Delay (min)'] <= 5
    return schedule_df, delay_df

schedule_df, delay_df = clean_data()

section = st.sidebar.radio("Go to section:", ["EDA", "Route Performance", "Delay Prediction"])

if section == "Delay Prediction":
    st.header("Train Delay Prediction")

    delay_model_df = delay_df.copy()
    label_encoders = {}
    input_features = ['Weather Conditions', 'Day of the Week', 'Time of Day', 'Train Type', 'Route Congestion']
    for col in input_features:
        le = LabelEncoder()
        delay_model_df[col] = le.fit_transform(delay_model_df[col])
        label_encoders[col] = le

    X = delay_model_df.drop(columns=['Historical Delay (min)', 'On_Time'])
    y = delay_model_df['Historical Delay (min)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.subheader("Enter Train Conditions")

    # ✅ Use 'Train_No' instead of 'Train Number'
    if 'Train_No' in info_df.columns:
        train_no_list = sorted(info_df['Train_No'].dropna().astype(str).unique())
        selected_train = st.selectbox("Select Train Number (optional)", ["Manual Entry"] + list(train_no_list))
    else:
        st.warning("Train_No column not found in train_info.csv")
        selected_train = "Manual Entry"

    default_type = None
    default_distance = 200

    if selected_train != "Manual Entry":
        try:
            train_data = info_df[info_df['Train_No'].astype(str) == selected_train].iloc[0]
            default_type = train_data.get('Type', None)

            distance_value = schedule_df[schedule_df['Train_No'].astype(str) == selected_train]['Distance'].max()
            default_distance = int(distance_value) if not pd.isna(distance_value) else 200
        except Exception as e:
            st.warning("Train details not found or error occurred.")
            default_type = None
            default_distance = 200

    col1, col2, col3 = st.columns(3)

    with col1:
        weather = st.selectbox("Weather Conditions", label_encoders['Weather Conditions'].classes_)
        day = st.selectbox("Day of the Week", label_encoders['Day of the Week'].classes_)

    with col2:
        time_of_day = st.selectbox("Time of Day", label_encoders['Time of Day'].classes_)
        train_type = st.selectbox("Train Type", label_encoders['Train Type'].classes_,
                                  index=list(label_encoders['Train Type'].classes_).index(default_type)
                                  if default_type in label_encoders['Train Type'].classes_ else 0)

    with col3:
        congestion = st.selectbox("Route Congestion", label_encoders['Route Congestion'].classes_)
        distance = st.slider("Distance (km)", 10, 1000, int(default_distance))

    if st.button("Predict Delay"):
        try:
            sample = {
                'Weather Conditions': label_encoders['Weather Conditions'].transform([weather])[0],
                'Day of the Week': label_encoders['Day of the Week'].transform([day])[0],
                'Time of Day': label_encoders['Time of Day'].transform([time_of_day])[0],
                'Train Type': label_encoders['Train Type'].transform([train_type])[0],
                'Route Congestion': label_encoders['Route Congestion'].transform([congestion])[0],
                'Distance Between Stations (km)': distance
            }

            sample_df = pd.DataFrame([sample])
            sample_df = sample_df[X.columns]

            prediction = model.predict(sample_df)[0]
            st.success(f"Predicted Delay: {round(prediction)} minutes")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


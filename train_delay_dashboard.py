# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:57:33 2025

@author: kavin
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image

st.set_page_config(page_title="Train Delay Dashboard", layout="wide")

cris_logo = Image.open("logo_cris.png")
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

if section == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Average Delay by Train Type")
    train_type_delay = delay_df.groupby('Train Type')['Historical Delay (min)'].mean().reset_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(data=train_type_delay, x='Train Type', y='Historical Delay (min)', ax=ax1, palette='Set2')
    ax1.set_title("Average Delay (min) by Train Type")
    st.pyplot(fig1)

    st.subheader("Average Delay by Day of the Week")
    ordered_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    delay_df['Day of the Week'] = pd.Categorical(delay_df['Day of the Week'], categories=ordered_days, ordered=True)
    daywise = delay_df.groupby('Day of the Week')['Historical Delay (min)'].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=daywise, x='Day of the Week', y='Historical Delay (min)', marker='o', ax=ax2)
    ax2.set_title("Avg Delay by Day")
    st.pyplot(fig2)

    st.subheader("On-Time Percentage by Train Type")
    ontime_by_type = delay_df.groupby('Train Type')['On_Time'].mean().reset_index()
    ontime_by_type['On_Time (%)'] = ontime_by_type['On_Time'] * 100
    fig3, ax3 = plt.subplots()
    sns.barplot(data=ontime_by_type, x='Train Type', y='On_Time (%)', ax=ax3, palette='YlGn')
    ax3.set_title("On-Time % by Train Type")
    ax3.set_ylim(0, 100)
    st.pyplot(fig3)

    st.subheader("On-Time Percentage by Day of the Week")
    ontime_by_day = delay_df.groupby('Day of the Week')['On_Time'].mean().reset_index()
    ontime_by_day['On_Time (%)'] = ontime_by_day['On_Time'] * 100
    fig4, ax4 = plt.subplots()
    sns.lineplot(data=ontime_by_day, x='Day of the Week', y='On_Time (%)', marker='o', color='green', ax=ax4)
    ax4.set_title("On-Time % by Day")
    ax4.set_ylim(0, 100)
    st.pyplot(fig4)

elif section == "Route Performance":
    st.header("Route Performance Metrics")

    st.subheader("Number of Stops per Train")
    stops = schedule_df.groupby('Train_No')['Station_Code'].count().reset_index(name='Num_Stops')
    top_stops = stops.sort_values(by='Num_Stops', ascending=False).head(10)
    fig5, ax5 = plt.subplots()
    sns.barplot(data=top_stops, x='Train_No', y='Num_Stops', ax=ax5, palette='Blues')
    ax5.set_title("Top 10 Trains with Most Stops")
    st.pyplot(fig5)

    st.subheader("Longest Distance Trains")
    distance_df = schedule_df.groupby('Train_No')['Distance'].last().reset_index()
    distance_df['Distance'] = pd.to_numeric(distance_df['Distance'], errors='coerce')
    top_distance = distance_df.sort_values(by='Distance', ascending=False).head(10)
    fig6, ax6 = plt.subplots()
    sns.barplot(data=top_distance, x='Train_No', y='Distance', ax=ax6, palette='Purples')
    ax6.set_title("Top 10 Longest Distance Trains")
    st.pyplot(fig6)

elif section == "Delay Prediction":
    st.header("Train Delay Prediction")

    delay_model_df = delay_df.copy()
    label_encoders = {}
    input_features = ['Weather Conditions', 'Day of the Week', 'Time of Day', 'Train Type', 'Route Congestion']
    for col in input_features:
        le = LabelEncoder()
        delay_model_df[col] = le.fit_transform(delay_model_df[col])
        label_encoders[col] = le

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X = delay_model_df.drop(columns=['Historical Delay (min)', 'On_Time'])
    y = delay_model_df['Historical Delay (min)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.subheader("Enter Train Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        weather = st.selectbox("Weather Conditions", label_encoders['Weather Conditions'].classes_)
        day = st.selectbox("Day of the Week", label_encoders['Day of the Week'].classes_)

    with col2:
        time_of_day = st.selectbox("Time of Day", label_encoders['Time of Day'].classes_)
        train_type = st.selectbox("Train Type", label_encoders['Train Type'].classes_)

    with col3:
        congestion = st.selectbox("Route Congestion", label_encoders['Route Congestion'].classes_)
        distance = st.slider("Distance (km)", 10, 1000, 200)

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


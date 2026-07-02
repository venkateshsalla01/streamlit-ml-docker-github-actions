import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

st.title("House Price Prediction")

MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Average Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict"):
    data = pd.DataFrame([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]],
                        columns=['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'])
    prediction = model.predict(data)
    st.success(f"Predicted House price: ${prediction[0]*100000:.2f}")
    st.balloons()


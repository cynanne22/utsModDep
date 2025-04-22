import streamlit as st
import pandas as pd
import pickle
import gzip
from predictor import BookingPredictor

st.title("Hotel Booking Status Predictor")

with gzip.open("random_forest_model.pkl.gz", "rb") as f:
    model = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    standard_scaler = pickle.load(f)

with open("robust_scaler.pkl", "rb") as f:
    robust_scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)


predictor = BookingPredictor(
    model_path=None,
    standard_scaler_path=None,
    robust_scaler_path=None,
    columns_path=None
)
predictor.model = model
predictor.standard_scaler = standard_scaler
predictor.robust_scaler = robust_scaler
predictor.columns = columns


case_A = {
    "Booking_ID": "INN00005",
    "no_of_adults": 2,
    "no_of_children": 0,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 1,
    "type_of_meal_plan": "Not Selected",
    "required_car_parking_space": 0,
    "room_type_reserved": "Room_Type 1",
    "lead_time": 48,
    "arrival_year": 2018,
    "arrival_month": 4,
    "arrival_date": 11,
    "market_segment_type": "Online",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 94.50,
    "no_of_special_requests": 0
}

case_B = {
    "Booking_ID": "INN36271",
    "no_of_adults": 3,
    "no_of_children": 0,
    "no_of_weekend_nights": 2,
    "no_of_week_nights": 6,
    "type_of_meal_plan": "Meal Plan 1",
    "required_car_parking_space": 0,
    "room_type_reserved": "Room_Type 4",
    "lead_time": 85,
    "arrival_year": 2018,
    "arrival_month": 8,
    "arrival_date": 3,
    "market_segment_type": "Online",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 167.80,
    "no_of_special_requests": 1
}


test_options = {
    "Case A": case_A,
    "Case B": case_B
}

option = st.selectbox("Test Case", list(test_options.keys()))
selected_data = test_options[option]

st.write("#### Selected Data:")
st.dataframe(pd.DataFrame([selected_data]))

if st.button("Predict"):
    input_df = pd.DataFrame([selected_data])
    prediction = predictor.predict(input_df)
    result = "Canceled" if prediction[0] == 1 else "Not Canceled"
    st.success(f"Hasil Prediksi: {result}")


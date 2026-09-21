import streamlit as st
import joblib
import pandas as pd

# Load the trained model
# Make sure 'logi.sav' is in the same directory as your app.py
logi = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Input features from X.columns
# Index(['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
#        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
#        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
#        'Warehouse_Processing_Time'],
#       dtype='object')

delivery_distance = st.number_input('Delivery Distance (e.g., 19.35)', min_value=0.0, value=19.0)
traffic_congestion = st.selectbox('Traffic Congestion (1-5)', options=[1, 2, 3, 4, 5])
weather_condition = st.selectbox('Weather Condition (1-3)', options=[1, 2, 3])
delivery_slot = st.selectbox('Delivery Slot (1-3)', options=[1, 2, 3])
driver_experience = st.number_input('Driver Experience (Years, e.g., 16)', min_value=0, value=10)
num_stops = st.number_input('Number of Stops (e.g., 6)', min_value=0, value=5)
vehicle_age = st.number_input('Vehicle Age (Years, e.g., 9)', min_value=0, value=5)
road_condition_score = st.selectbox('Road Condition Score (1-4)', options=[1, 2, 3, 4])
package_weight = st.number_input('Package Weight (e.g., 33.62)', min_value=0.0, value=15.0)
fuel_efficiency = st.number_input('Fuel Efficiency (e.g., 13.01)', min_value=0.0, value=10.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (Minutes, e.g., 58)', min_value=0, value=60)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = logi.predict(input_data)
    prediction_proba = logi.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f"The model predicts a **Delivery Delay**.")
    else:
        st.success(f"The model predicts **No Delivery Delay**.")
    
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")

st.markdown("""
To run this app locally:
1. Save the code above into a file named `app.py`.
2. Ensure you have `streamlit`, `pandas`, and `scikit-learn` installed (`pip install streamlit pandas scikit-learn`).
3. Make sure the `logi.sav` file is in the same directory as `app.py`.
4. Open your terminal or command prompt, navigate to the directory where you saved `app.py`, and run `streamlit run app.py`.
""")

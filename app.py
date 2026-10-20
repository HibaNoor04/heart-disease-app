import streamlit as st
import pandas as pd
import pickle

# Load your trained model
# Note: Make sure the file name matches exactly what your notebook saved!
model = pickle.load(open('svc_trained_model.pkl', 'rb'))

st.title("Heart Disease Prediction System")
st.write("Enter the patient's details below to predict the likelihood of heart disease.")

# 1. Get user inputs matching your 4 attributes
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", options=[0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=220, value=150)

# 2. Convert the inputs into a DataFrame
if st.button("Predict"):
    user_input = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'thalach': [thalach]
    })
    
    # 3. Make the prediction
    prediction = model.predict(user_input)
    
    # 4. Display the result
    if prediction[0] == 1:
        st.error("Prediction: DISEASE DETECTED")
    else:
        st.success("Prediction: HEALTHY (No Disease Detected)")
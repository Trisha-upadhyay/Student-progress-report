import streamlit as st
import joblib

model = joblib.load("model.pkl")

st.title("Student Performance Prediction System")

attendance = st.slider("Attendance (%)",0,100,70)
study = st.slider("Study Hours",0,10,3)
assign = st.slider("Assignments Score",0,100,60)
lms = st.slider("LMS Logins",0,20,8)

if st.button("Predict"):

    prediction = model.predict([[attendance,study,assign,lms]])

    st.write("Predicted Grade:",prediction[0])

    if prediction < 50:
        st.error("⚠ Student is at risk")
    else:
        st.success("Student performance is good")
import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

# ---------- LOGIN SYSTEM ----------
def login():
    st.title("🔐 Login System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "trisha" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")


# ---------- MAIN DASHBOARD ----------
def dashboard():
    st.title("📊 Student Performance Prediction System")

    st.write("Enter student details")

    attendance = st.slider("Attendance (%)", 0, 100, 70)
    study = st.slider("Study Hours", 0, 10, 3)
    assign = st.slider("Assignment Score", 0, 100, 60)
    lms = st.slider("LMS Logins", 0, 20, 8)

    if st.button("Predict"):

        pred = model.predict([[attendance, study, assign, lms]])[0]

        st.write("Predicted Grade:", round(pred, 2))

        # Correct Risk Logic
        if pred < 30:
            st.error("🔴 High Risk Student")
        elif pred < 60:
            st.warning("🟠 Medium Risk Student")
        else:
            st.success("🟢 Low Risk Student")

        # Extra smart condition
        if attendance == 0 and study == 0:
            st.error("⚠ Extreme Risk: No attendance & no study")


# ---------- MAIN APP ----------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    dashboard()
else:
    login()
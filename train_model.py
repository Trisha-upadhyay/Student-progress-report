import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("student_data.csv")

X = data[['Attendance','Study_Hours','Assignments','LMS_Logins']]
y = data['Final_Grade']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor()
model.fit(X_train,y_train)

joblib.dump(model,"model.pkl")

print("Model trained successfully")
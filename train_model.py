import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {
    "attendance": [10,20,30,40,50,60,70,80,90],
    "study":      [1,2,2,3,3,4,5,6,7],
    "assign":     [20,30,40,50,60,70,80,90,100],
    "lms":        [1,2,3,4,5,6,7,8,9],
    "grade":      [20,30,40,50,60,70,80,90,95]
}

df = pd.DataFrame(data)

X = df[["attendance","study","assign","lms"]]
y = df["grade"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully!")
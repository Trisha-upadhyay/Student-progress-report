from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    pred = model.predict([[ 
        float(data["attendance"]), 
        float(data["study"]), 
        float(data["assign"]), 
        float(data["lms"]) 
    ]])[0]

    if pred < 30:
        risk = "High"
    elif pred < 60:
        risk = "Medium"
    else:
        risk = "Low"

    return jsonify({
        "score": round(float(pred), 2),
        "risk": risk
    })

if __name__ == "__main__":
    app.run(debug=True)
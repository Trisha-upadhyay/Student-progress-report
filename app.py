from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import joblib
import json
import os
import pandas as pd
import sqlite3
from urllib import error
from urllib import request as urllib_request

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency for local env loading.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_FEATURES = ["attendance", "study", "assign", "lms"]
model = joblib.load("model.pkl")


def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attendance REAL NOT NULL,
            study REAL NOT NULL,
            assign_score REAL NOT NULL,
            lms REAL NOT NULL,
            score REAL NOT NULL,
            risk TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Backward-compatible migration for old databases.
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row["name"] for row in cursor.fetchall()}
    if "ai_provider" not in existing_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN ai_provider TEXT")
    if "ai_suggestion" not in existing_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN ai_suggestion TEXT")

    cursor.execute("SELECT id FROM users WHERE email = ?", ("admin@gmail.com",))
    user = cursor.fetchone()
    if not user:
        cursor.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            ("admin@gmail.com", "1234"),
        )

    conn.commit()
    conn.close()


def classify_risk(score):
    if score < 30:
        return "High"
    if score < 60:
        return "Medium"
    return "Low"


def build_prompt(data, score, risk):
    return (
        "You are an academic advisor. Give 2-3 short actionable suggestions.\n"
        f"Attendance: {data['attendance']}\n"
        f"Study hours: {data['study']}\n"
        f"Assignment score: {data['assign']}\n"
        f"LMS activity: {data['lms']}\n"
        f"Predicted score: {round(score, 2)}\n"
        f"Risk level: {risk}\n"
        "Focus on practical next steps."
    )


def request_chat_completion(api_url, api_key, model_name, prompt, extra_headers=None):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a concise academic performance coach."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 180,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    req = urllib_request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=12) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


def get_local_suggestion(data, risk):
    suggestions = []
    if data["attendance"] < 75:
        suggestions.append("Increase attendance to at least 75% by planning a weekly class routine.")
    if data["study"] < 4:
        suggestions.append("Raise study time to at least 4 focused hours per day with a fixed timetable.")
    if data["assign"] < 70:
        suggestions.append("Submit assignments early and review rubric requirements before final submission.")
    if data["lms"] < 6:
        suggestions.append("Log into LMS daily to track announcements, deadlines, and revision resources.")
    if not suggestions:
        suggestions.append("Maintain current habits and run weekly self-checks to keep performance stable.")
    if risk == "High":
        suggestions.append("Meet a mentor weekly until the risk level improves.")
    return " ".join(suggestions[:3])


def generate_ai_suggestion(data, score, risk):
    prompt = build_prompt(data, score, risk)

    providers = [
        {
            "name": "Groq",
            "key": os.getenv("GROQ_API_KEY"),
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "headers": None,
        },
        {
            "name": "OpenRouter",
            "key": os.getenv("OPENROUTER_API_KEY"),
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
            "headers": {"HTTP-Referer": "http://localhost:5000", "X-Title": "Student AI Dashboard"},
        },
    ]

    for provider in providers:
        if not provider["key"]:
            continue
        try:
            text = request_chat_completion(
                provider["url"],
                provider["key"],
                provider["model"],
                prompt,
                provider["headers"],
            )
            if text:
                return provider["name"], text
        except (KeyError, error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            continue

    return "Local", get_local_suggestion(data, risk)


def parse_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    parsed = {}
    for field in MODEL_FEATURES:
        value = payload.get(field)
        if value is None:
            raise ValueError(f"Missing required field: {field}")
        try:
            parsed[field] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {field}. Must be numeric.") from exc

    return parsed


@app.route("/")
def home():
    return render_template("login.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE email = ? AND password = ?",
            (username, password),
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            return redirect(url_for("dashboard"))
        return "Invalid username or password", 401

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = parse_payload(request.get_json(silent=True))
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    try:
        # DataFrame with feature names avoids sklearn warning.
        model_input = pd.DataFrame([data], columns=MODEL_FEATURES)
        score = float(model.predict(model_input)[0])
        score = max(0.0, min(100.0, score))
        risk = classify_risk(score)
        ai_provider, ai_suggestion = generate_ai_suggestion(data, score, risk)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions
            (attendance, study, assign_score, lms, score, risk, ai_provider, ai_suggestion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["attendance"],
                data["study"],
                data["assign"],
                data["lms"],
                score,
                risk,
                ai_provider,
                ai_suggestion,
            ),
        )
        conn.commit()
        conn.close()

        return jsonify(
            {
                "score": round(score, 2),
                "risk": risk,
                "ai_provider": ai_provider,
                "ai_suggestion": ai_suggestion,
            }
        )
    except Exception:
        return jsonify({"error": "Prediction failed due to a server error."}), 500


@app.route("/predictions/history", methods=["GET"])
def predictions_history():
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit, 100))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, attendance, study, assign_score, lms, score, risk, ai_provider, ai_suggestion, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    history = [dict(row) for row in rows]
    return jsonify({"count": len(history), "history": history})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)

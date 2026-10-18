
import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "news_classifier_model.pkl")
model = joblib.load(MODEL_PATH)

CATEGORIES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

@app.route("/", methods=["GET"])
def home():
    return "NLP News Category Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    headline = data.get("headline", "").strip()

    if not headline:
        return jsonify({"error": "Please enter a news headline"}), 400

    prediction = int(model.predict([headline])[0])
    category = CATEGORIES.get(prediction, "Unknown")

    return jsonify({
        "headline": headline,
        "category": category
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

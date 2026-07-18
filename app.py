"""
app.py
======

Simple Flask web app for the Phishing URL Detection college project.

How it works:
    1. User enters a URL on the home page (index.html).
    2. We extract features from that URL using features.py.
    3. We load the pre-trained model (model.pkl) and predict.
    4. We show the result (Legitimate / Phishing) on result.html.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request
import joblib
import pandas as pd

from features import extract_features, FeatureExtractionError

app = Flask(__name__)

# ------------------------------------------------------------------
# Load the trained model ONCE when the app starts (not on every request)
# ------------------------------------------------------------------
model = joblib.load("model.pkl")

# The exact order of feature columns the model was trained on.
# We MUST send the features to the model in this exact order.
FEATURE_ORDER = list(model.feature_names_in_)


@app.route("/")
def home():
    """Show the input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle the form submission: extract features, predict, show result."""
    url = request.form.get("url", "").strip()

    if not url:
        return render_template("result.html", error="Please enter a URL.")

    try:
        # Step 1: turn the URL into the 51 numeric features the model needs
        feature_dict = extract_features(url)

        # Step 2: arrange them in a table (DataFrame) with the correct
        # column order, since that is what the model expects
        feature_row = pd.DataFrame([feature_dict])[FEATURE_ORDER]

        # Step 3: ask the model to predict
        # (In this dataset, label 1 = Legitimate, label 0 = Phishing)
        prediction = model.predict(feature_row)[0]
        result = "Legitimate" if prediction == 1 else "Phishing"

        return render_template("result.html", url=url, result=result)

    except FeatureExtractionError as exc:
        # Friendly error if we could not reach/read the website
        return render_template("result.html", url=url, error=str(exc))

    except Exception as exc:  # noqa: BLE001 - keep it simple for a college project
        return render_template(
            "result.html", url=url, error=f"Something went wrong: {exc}"
        )


if __name__ == "__main__":
    app.run(debug=True)

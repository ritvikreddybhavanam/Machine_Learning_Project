
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Allow requests from your deployed website.
# For production, you can restrict this to your website's URL.
CORS(app)

# Find the repository root from src/backend/app.py
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR, "src", "models", "final_elastic_net_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "Dataset", "processed", "cleaned_movie_data.csv"
)

# Load the model and dataset
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

categorical_columns = [
    "Release_Period",
    "Whether_Remake",
    "Whether_Franchise",
    "Genre",
    "New_Actor",
    "New_Director",
    "New_Music_Director",
    "Lead_Star",
    "Director",
    "Music_Director"
]

# Recreate the training feature columns
training_features = df.drop(
    columns=["Movie_Name", "Revenue(INR)"]
)

training_encoded = pd.get_dummies(
    training_features,
    columns=categorical_columns,
    drop_first=True
)

training_encoded = training_encoded.astype(float)
feature_columns = training_encoded.columns.tolist()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Box Office Prediction API is running",
        "prediction_endpoint": "/api/predict",
        "method": "POST"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "model_loaded": model is not None,
        "dataset_loaded": not df.empty
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Request body must be valid JSON."
            }), 400

        required_fields = [
            "releasePeriod",
            "whetherRemake",
            "whetherFranchise",
            "genre",
            "newActor",
            "newDirector",
            "newMusicDirector",
            "leadStar",
            "director",
            "musicDirector",
            "numberOfScreens",
            "budget"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data or data[field] is None
            or (isinstance(data[field], str) and not data[field].strip())
        ]

        if missing_fields:
            return jsonify({
                "success": False,
                "error": "Missing required fields.",
                "missingFields": missing_fields
            }), 400

        try:
            number_of_screens = float(data["numberOfScreens"])
            budget = float(data["budget"])
        except (TypeError, ValueError):
            return jsonify({
                "success": False,
                "error": "numberOfScreens and budget must be numbers."
            }), 400

        if (
            not pd.notna(number_of_screens)
            or not pd.notna(budget)
            or number_of_screens <= 0
            or budget < 0
        ):
            return jsonify({
                "success": False,
                "error": "Screens must be positive and budget cannot be negative."
            }), 400

        movie_data = pd.DataFrame([{
            "Release_Period": data["releasePeriod"],
            "Whether_Remake": data["whetherRemake"],
            "Whether_Franchise": data["whetherFranchise"],
            "Genre": data["genre"],
            "New_Actor": data["newActor"],
            "New_Director": data["newDirector"],
            "New_Music_Director": data["newMusicDirector"],
            "Lead_Star": data["leadStar"],
            "Director": data["director"],
            "Music_Director": data["musicDirector"],
            "Number_of_Screens": number_of_screens,
            "Budget(INR)": budget
        }])

        movie_encoded = pd.get_dummies(
            movie_data,
            columns=categorical_columns,
            drop_first=True
        )

        # Align columns with the model's training features
        movie_encoded = movie_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        ).astype(float)

        prediction = float(model.predict(movie_encoded)[0])

        if not pd.notna(prediction):
            raise ValueError("The model returned an invalid prediction.")

        prediction = max(0.0, prediction)

        return jsonify({
            "success": True,
            "predictedRevenue": prediction,
            "currency": "INR"
        }), 200

    except Exception as error:
        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


if __name__ == "__main__":
    # Render supplies the PORT environment variable.
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

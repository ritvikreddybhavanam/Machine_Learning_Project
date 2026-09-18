from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import joblib
import os


app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "models",
    "final_elastic_net_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "processed",
    "cleaned_movie_data.csv"
)


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


@app.route("/")
def home():
    return jsonify({
        "message": "Box Office Prediction API is running"
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

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
            "Number_of_Screens": float(data["numberOfScreens"]),
            "Budget(INR)": float(data["budget"])
        }])


        movie_encoded = pd.get_dummies(
            movie_data,
            columns=categorical_columns,
            drop_first=True
        )


        movie_encoded = movie_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )


        movie_encoded = movie_encoded.astype(float)


        prediction = model.predict(movie_encoded)[0]


        if prediction < 0:
            prediction = 0


        return jsonify({
            "success": True,
            "predictedRevenue": float(prediction)
        })


    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
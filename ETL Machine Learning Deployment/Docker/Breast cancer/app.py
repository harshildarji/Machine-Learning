import pandas as pd
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    input_data = req["data"]
    input_data_frame = pd.DataFrame.from_dict(input_data)

    model = pickle.load(open("svc_model.pkl", "rb"))
    pred = model.predict(input_data_frame)

    if pred[0] == 1:
        c_type = "Malignant"
    else:
        c_type = "Benign"

    return jsonify({"output": {"cancer_type": c_type}})


@app.route("/")
def home():
    return "Breast Cancer Prediction"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")

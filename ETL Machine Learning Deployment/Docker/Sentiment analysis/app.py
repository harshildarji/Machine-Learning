from flask import Flask, render_template, request
import jsonify
import requests
import pickle

import warnings
warnings.simplefilter('ignore')


app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

sentiment = {
    1: "Positive",
    0: "Negative"
}


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if request.method == "POST":
        review = str(request.form["testsentence"]).strip()

        if review:
            pred = int(model.predict(vectorizer.transform([review])))
            if pred == 1:
                output = '<b style="color:#009900">{}</b>'.format(sentiment[pred])
            elif pred == 0:
                output = '<b style="color:#990000">{}</b>'.format(sentiment[pred])
            else:
                output = '<b style="color:#999900">Something went really wrong!</b>'

            output = "<strong>- Original review -</strong><br><br>{}<br><br><strong>- Sentiment -</strong><br><br><h3>{}</h3>".format(
                review, output
            )
        else:
            output = '<b style="color:#999900">Can\'t find sentiment of an EMPTY string.</b>'

        return render_template("index.html", analyzed_sentence="{}".format(output))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")

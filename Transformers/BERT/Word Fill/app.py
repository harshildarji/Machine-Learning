import warnings

import torch
from flask import Flask, render_template, request
from transformers import AutoModelForMaskedLM, BertTokenizer, logging

warnings.simplefilter(action="ignore", category=Warning)
logging.set_verbosity(logging.ERROR)


print("[+] this might take a few minutes...")

app = Flask(__name__)
device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")


@app.route("/fill", methods=["POST"])
def fill():
    if request.method == "POST":
        test_sentence = str(request.form["testsentence"])

        split_text = test_sentence.split("--")
        text = test_sentence.replace("--", "[MASK] ")
        text = "[CLS] " + text + " [SEP]"

        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = [i for i, x in enumerate(tokenized_text) if x == "[MASK]"]

        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)

        for i, mi in enumerate(masked_index):
            predicted_index = torch.argmax(predictions[0][0, mi]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            split_text[
                i
            ] += f'<t style="color:#1862f7"><strong>{predicted_token}</strong></t>'

        output = f"{''.join(split_text)}"
        output = "<mark><strong>Original text</strong></mark><br><br>{}<br><br><mark><strong>Filled in text</strong></mark><br><br>{}".format(
            test_sentence, output
        )
        return render_template("index.html", filled_sentence="{}".format(output))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5050")

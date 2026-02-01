"""

"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from model.modified_network import NeuralNetwork
from model.modified_layers import *
import numpy as np
import io

app = Flask(__name__)
CORS(app)


LAYERS = [
        Conv2D(3, 16, 3, stride=1, padding=1),
        MaxPool2D(2, 2),

        Conv2D(16, 32, 3, stride=1, padding=1),
        MaxPool2D(2, 2),

        Conv2D(32, 64, 3, stride=1, padding=1),
        MaxPool2D(2, 2),

        Flatten(),
        DenseLayer(16384, 128, False),
        DenseLayer(128, 2, True)
    ]

model = NeuralNetwork(LAYERS)
model.load("model/model.npz")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    img = np.array(img, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...]

    output = model.forward_propagation(img)

    prediction = int(np.argmax(output))
    confidence = float(np.max(output))

    label = "Benign" if prediction == 0 else "Malignant"

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)

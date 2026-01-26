"""
TODO:
    Deployment Steps:
    Make repo with backend: app.py and requirements.txt
    Go to https://render.com
    Login with GitHub
    Click New -> Web Service
    Select the backend repo
    Fill in:
        -Runtime: Python
        -Start command:
            gunicorn app:app
        -Environment: Python 3

    Use given URL in API_URL in script.js

    What still needs to be done after this:
        -Find an optimal model (hyperparameters)
        -Save/Load model
        -Change this file, so we are calling the model and not dummy
        -Push to backend repo and update Render
        -Test/END
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)


# FAKE MODEL FOR DEMONSTRATION
def dummy_model_forward(x):
    # Pretend output: [benign_prob, malignant_prob]
    return np.array([[0.3], [0.7]])


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    img = img.resize((128, 128))

    img = np.array(img, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...]

    # CHANGE THIS WHEN ACTUAL MODEL WORKS
    output = dummy_model_forward(img)
    # output = model.forward_propagation(img)

    prediction = int(np.argmax(output))
    confidence = float(np.max(output))

    label = "benign" if prediction == 0 else "malignant"

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)

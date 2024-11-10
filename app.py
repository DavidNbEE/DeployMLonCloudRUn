from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS  # Tambahkan import CORS
import numpy as np
import json
import os
from PIL import Image

app = Flask(__name__)
CORS(app)  # Inisialisasi CORS

# Load model dan class_indices
MODEL_PATH = 'trained_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'
model = load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
classes = list(class_indices.keys())

# Fungsi untuk preprocess gambar
def preprocess_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi
    return img_array

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Baca dan preprocess gambar
        img = Image.open(file.stream)
        img_array = preprocess_image(img)

        # Prediksi menggunakan model
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        class_name = classes[class_idx]
        confidence = float(prediction[0][class_idx])

        # Return hasil prediksi sebagai JSON
        return jsonify({"predicted_class": class_name, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Menjalankan API
if __name__ == '__main__':
    app.run(debug=True)

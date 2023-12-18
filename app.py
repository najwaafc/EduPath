from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load data
df = pd.read_excel("Data_Jurusan_Lengkap.xlsx")

label_encoder = LabelEncoder()
label_encoder.fit(df['Jurusan'])

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Deskripsi'])

# Load the pre-trained model
model = load_model("model_1.h5")  # Ganti dengan path yang sesuai dengan model Anda

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Missing 'text' input in request body."
                },
                "data": None
            }), 400

        user_input = data["text"]

        # Pastikan teks yang lebih panjang diproses dengan benar
        user_input_vector = tfidf_vectorizer.transform([user_input])

        # Reorder vektor sparse
        user_input_vector_reordered = user_input_vector.tocsr()
        user_input_vector_reordered.sort_indices()

        predicted_probabilities = model.predict(user_input_vector_reordered)
        
        # Menangani kasus input yang tidak dapat diprediksi
        if not predicted_probabilities.any():
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Unable to make a prediction for the provided text."
                },
                "data": {
                    "input_text": user_input
                }
            }), 400

        predicted_class = np.argmax(predicted_probabilities)
        predicted_jurusan = label_encoder.inverse_transform([predicted_class])[0]

        return jsonify({
            "status": {
                "code": 200,
                "message": "Prediction successful."
            },
            "data": {
                "input_text": user_input,
                "predicted_class": predicted_jurusan
            }
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed. Use POST request for predictions."
            },
            "data": None
        }), 405

if __name__ == "__main__":
    app.run()

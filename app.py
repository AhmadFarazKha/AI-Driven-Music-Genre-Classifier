from flask import Flask, request, jsonify
import joblib
import numpy as np
from feature_extraction import extract_features, get_file_extension
import os

app = Flask(__name__)
model = joblib.load("genre_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    temp_file_path = "temp" + get_file_extension(audio_file.filename)
    audio_file.save(temp_file_path)
    features = extract_features(temp_file_path)
    os.remove(temp_file_path) #remove the temp file.

    if features is None:
        return jsonify({'error': 'Error processing audio file'}), 500
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    return jsonify({'genre': prediction})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import numpy as np
import requests
import tensorflow as tf
import cv2
import base64


# Decode the base64 image
def decode_base64_to_array(base64_image):
    return cv2.imdecode(np.frombuffer(base64.b64decode(base64_image), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


# Covert the image to grayscale
def to_grayscale(image_arr):
    return cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)


# Initialize Flask
app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            req_json = request.get_json()
            image = ""

            # Convert the request from list to string
            for letter in req_json:
                image += letter

            image_decoded = decode_base64_to_array(image)

            # Get image grayscale
            gray = to_grayscale(image_decoded)

            # Reshape the array so it fits the model
            query = gray.reshape(1, 400)

            prediction = model.predict(query)
            prediction_softmax = tf.nn.softmax(prediction)
            final_prediction = int(np.argmax(prediction_softmax))

            return {'prediction': final_prediction}


        except:
            return jsonify({'trace': traceback.format_exc()})

    else:
        return "Model error."


model = tf.keras.models.load_model('saved_model/recon_updated')

if __name__ == "__main__":
    app.run(port=0000)

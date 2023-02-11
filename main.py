from flask import Flask, request, jsonify
import joblib
import traceback
import numpy as np
import requests
import tensorflow as tf

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            req_json = list(request.get_json())
            print(req_json)
            req_json = np.array(req_json)
            query = req_json.reshape(1, 400)

            prediction = model.predict(query)
            prediction_softmax = tf.nn.softmax(prediction)
            final_prediction = int(np.argmax(prediction_softmax))

            print(final_prediction)

            return {'prediction': final_prediction}


        except:
            return jsonify({'trace': traceback.format_exc()})

    else:
        return "Model error."


model = joblib.load("new_recon.pkl")

if __name__ == "__main__":
    app.run(debug=True)


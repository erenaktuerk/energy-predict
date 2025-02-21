from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('./model/energy_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict energy consumption based on input temperature.

    Expected JSON format:
    {
        "temperature": float
    }
    """
    data = request.get_json()
    temperature = data.get('temperature')

    if temperature is None:
        return jsonify({"error": "Temperature is required"}), 400

    # Make prediction
    prediction = model.predict([[temperature]])
    return jsonify({"predicted_energy_consumption": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
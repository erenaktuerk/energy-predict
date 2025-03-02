from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Path to the trained model
MODEL_PATH = os.path.join('model', 'energy_model.pkl')

# Define expected features manually
expected_features = ['temperature', 'population', 'gdp', 'biofuel_cons_change_pct']

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully.")
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}")
except AttributeError:
    model = None
    print("Error: Could not retrieve feature names from the model. Check model compatibility.")

@app.route('/')
def home():
    """
    Home route providing a simple welcome message.
    """
    return (
        "<h1>Energy Predict API</h1>"
        "<p>This API predicts energy consumption based on input features.</p>"
        "<p>Use the endpoint <code>/predict</code> with a POST request to get predictions.</p>"
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict energy consumption based on input features.

    Expected JSON format:
    {
        "temperature": float,
        "population": float,
        "gdp": float
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the model path."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Check if all required features are provided
    missing_features = [feature for feature in expected_features if feature not in data]
    if missing_features:
        return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

    try:
        # Extract and order features according to the model’s expected input
        features = [[float(data[feature]) for feature in expected_features]]
        prediction = model.predict(features)
        
        # ensure prediction is a standard python float (and not numpy float32)
        prediction_value = float(prediction[0])
        
        return jsonify({"predicted_energy_consumption": prediction_value})

    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Define the path to the trained model
MODEL_PATH = os.path.join('model', 'energy_model.pkl')

# Manually define the expected features (must match training)
# Only using the 4 features, we used in the training
expected_features = ['year', 'gdp_per_capita', 'population', 'energy_per_capita']

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Error loading model: {str(e)}")

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
        "year": float,
        "gdp_per_capita": float,
        "population": float,
        "energy_per_capita": float
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the model path."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Check for missing required features
    missing_features = [feature for feature in expected_features if feature not in data]
    if missing_features:
        return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

    try:
        # Order features as expected by the model
        features = [[float(data[feature]) for feature in expected_features]]
        prediction = model.predict(features)
        # Convert prediction to native float to ensure JSON serializability
        prediction_value = float(prediction[0])
        return jsonify({"predicted_energy_consumption": prediction_value})
    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)
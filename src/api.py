"""
Endpoints:
- GET  /health --> Service health check
- POST /predict --> Predict disease progression score
- GET  /metrics --> Model performance metrics
"""

import json
from pathlib import Path
import pandas as pd

import joblib
import numpy as np
from flask import Flask, request, jsonify


# Configuration

MODEL_PATH = Path('models/model.pkl')
METRICS_PATH = Path('models/metrics.json')
TRAINING_INFO_PATH = Path('models/training_info.json')

# Expected feature names (must match sklearn diabetes dataset order)
EXPECTED_FEATURES = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# API Setup
print('DIABETES TRIAGE API - STARTING UP')
print('=' * 60)

# Load the trained pipeline
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}.")

model_pipeline = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# Load metrics (for /metrics endpoint)
if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        model_metrics = json.load(f)
    print(f"Metrics loaded: RMSE={model_metrics['rmse']:.2f}")
else:
    model_metrics = {'error': 'Metrics file not found'}
    print('Warning! metrics.json not found')

# Load training info (for /health endpoint)
if TRAINING_INFO_PATH.exists():
    with open(TRAINING_INFO_PATH) as f:
        training_info = json.load(f)
    print(f"Training info loaded: {training_info['model_version']}")
else:
    training_info = {'model_version': 'unknown'}
    print('Warning! training_info.json not found')

print("=" * 60)
print("API READY - Listening on port 5000")
print("=" * 60)

# Create Flask App
app = Flask(__name__)

# Step 1: Health Check of the Model
@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for container orchestration.
    
    Returns:
        200: Service is healthy
        Example: {"status": "ok", "model_version": "v0.1"}
    """
    return jsonify({
        'status': 'ok',
        'model_version': training_info.get('model_version', 'unknown'),
        'model_type': training_info.get('model_type', 'unknown')
    }), 200


# Step 2: Predict Disease Progression
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diabetes progression score from patient features.
    
    Returns:
        200: {"prediction": 150.5}
        400: {"error": "Missing required fields: ['age', 'bmi']"}
        500: {"error": "Prediction failed: ..."}
    """
    try:
        # 1. Parsing incoming JSON
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        # 2. Validate if all required features are present
        missing_features = [f for f in EXPECTED_FEATURES if f not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing required fields: {missing_features}"
            }), 400
        
        # 3. Extract features in the correct order (as seen in the training)
        features = [data[f] for f in EXPECTED_FEATURES]
        
        # 4. Convert to pandas DataFrame with feature names
        X = pd.DataFrame([features], columns=EXPECTED_FEATURES)
        
        # 5. Validate feature types (should be numeric)
        if not np.isfinite(X.values).all():
            return jsonify({
                'error': 'All features must be finite numbers (no NaN/Inf)'
            }), 400
        
        # 6. Make prediction
        prediction = model_pipeline.predict(X)[0]  # Returns array
        
        # 7. Return result
        return jsonify({
            'prediction': float(prediction)
        }), 200
    
    except Exception as e:
        # Fail-safe mechanisms + explaining the error
        print(f"ERROR during prediction: {e}") 
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# Step 3: Model Metrics
@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Return model performance metrics.
    
    Returns:
        200: {"rmse": 56.4, "mae": 45.2, "r2": 0.45, ...}
    """
    return jsonify(model_metrics), 200


# Run the App
if __name__ == '__main__':
    # host='0.0.0.0' makes it accessible from outside of the container
    # port=5000 is the standard port
    # debug=False in production (no auto-reload, no debug info leakage)
    app.run(host='0.0.0.0', port=5000, debug=False)
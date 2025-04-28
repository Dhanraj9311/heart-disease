import numpy as np
from flask import Flask, request, jsonify
import pickle
import logging
from typing import Dict, Union, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model outside the route to avoid reloading on every request
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    # Fallback model (important for error handling)
    model = lambda x: np.array([0])  # A dummy model that always predicts 0


@app.route('/')
def home() -> str:
    """Home endpoint that returns a simple greeting."""
    return '<h1>\prediction</h1>'


@app.route('/prediction', methods=['POST'])
def prediction() -> Dict[str, Any]:
    """
    Prediction endpoint that takes heart disease parameters and returns a prediction.
    
    Returns:
        JSON response with prediction or error message
    """
    try:
        # List of required parameters
        required_params = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal'
        ]
        
        # Get data from the form
        input_data = {}
        for param in required_params:
            value = request.form.get(param)
            if value is None:
                return jsonify({
                    'error': f'Missing parameter: {param}',
                    'required_parameters': required_params
                }), 400
            input_data[param] = value

        # Convert data to float, handling potential errors
        try:
            input_values = [float(input_data[param]) for param in required_params]
        except ValueError as ve:
            return jsonify({
                'error': 'Invalid input data. Please ensure all values are numeric.',
                'details': str(ve)
            }), 400

        # Convert to numpy array with correct shape and type
        input_query = np.array([input_values], dtype=np.float64)

        # Make prediction
        result = model.predict(input_query)[0]
        
        # Return prediction
        return jsonify({
            'heart_disease': int(result),
            'status': 'success'
        })

    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred during prediction',
            'status': 'error'
        }), 500


if __name__ == '__main__':
    # Note: In production, debug should be False
    app.run(debug=True, host='0.0.0.0', port=5000)

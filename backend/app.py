from flask import Flask, jsonify, request
from pymongo import MongoClient
from kafka_producer import send_prediction_to_kafka
import numpy as np
import joblib
from tensorflow import keras
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MongoDB configuration
mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://username:password@cluster.mongodb.net/")
client = MongoClient(mongo_uri)
db = client.StockData

# Load Models and Scalers only once during app startup
models = {}


def load_models(company_id):
    """Load models and scalers for a given company."""
    if company_id not in models:
        try:
            lstm_model = keras.models.load_model(f'models/LSTM/{company_id}.h5')
            svr_model = joblib.load(f'models/SVR/{company_id}.pkl')
            xgbr_model = joblib.load(f'models/XGBR/{company_id}.pkl')
            scaler = joblib.load(f'models/Scalers/{company_id}.pkl')
            ensemble_model = joblib.load(f'models/Ensemble/{company_id}.pkl')
            models[company_id] = (lstm_model, svr_model, xgbr_model, scaler, ensemble_model)
            logger.info(f"Models for company {company_id} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models for company {company_id}: {e}")
            raise
    return models[company_id]


# Prediction logic
def predict_stock_price(data, company_id):
    """Predict stock price using LSTM, SVR, XGBoost, and combine via ensemble."""
    lstm_model, svr_model, xgbr_model, scaler, ensemble_model = load_models(company_id)

    try:
        # Scale input data
        scaled_data = scaler.transform(np.array(data).reshape(-1, 1))

        # LSTM Prediction
        lstm_pred = lstm_model.predict(np.reshape(scaled_data, (1, 1, len(data))))[0][0]

        # SVR Prediction
        svr_pred = svr_model.predict(scaled_data[-1:])[0]

        # XGBoost Prediction
        xgbr_pred = xgbr_model.predict(scaled_data[-1:])[0]

        # Ensemble Prediction
        ensemble_input = np.array([[lstm_pred, svr_pred, xgbr_pred]])
        ensemble_pred = ensemble_model.predict(ensemble_input)

        return {
            "lstm": lstm_pred,
            "svr": svr_pred,
            "xgboost": xgbr_pred,
            "ensemble": ensemble_pred[0]
        }
    except Exception as e:
        logger.error(f"Error during prediction for company {company_id}: {e}")
        return {"error": str(e)}


# API Endpoints
@app.route('/predictStockData', methods=['POST'])
def predict_stock_data():
    """Endpoint to predict stock data."""
    data = request.json

    if 'company_id' not in data or 'historical_data' not in data:
        logger.error("Invalid input data, missing company_id or historical_data.")
        return jsonify({"error": "Missing 'company_id' or 'historical_data' in the request."}), 400

    company_id = data['company_id']
    historical_data = data['historical_data']

    # Predict stock price
    predictions = predict_stock_price(historical_data, company_id)

    if 'error' in predictions:
        return jsonify(predictions), 500

    # Send prediction to Kafka
    send_prediction_to_kafka(company_id, predictions)

    return jsonify(predictions), 200


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Healthcheck endpoint."""
    return jsonify({"status": "Running"}), 200


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

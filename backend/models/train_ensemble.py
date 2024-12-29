import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load predictions from individual models
def load_predictions(company_id):
    try:
        lstm_preds = np.load(f'outputs/lstm_predictions/{company_id}_preds.npy')
        svr_preds = np.load(f'outputs/svr_predictions/{company_id}_preds.npy')
        xgboost_preds = np.load(f'outputs/xgboost_predictions/{company_id}_preds.npy')
        actual_values = np.load(f'outputs/actual_values/{company_id}_actual.npy')

        logger.info(f"Loaded predictions for company_id: {company_id}")
        return lstm_preds, svr_preds, xgboost_preds, actual_values
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        raise


def train_ensemble(company_id):
    lstm_preds, svr_preds, xgboost_preds, actual_values = load_predictions(company_id)

    # Check that all predictions have the same length
    if not (len(lstm_preds) == len(svr_preds) == len(xgboost_preds) == len(actual_values)):
        logger.error("Predictions and actual values have mismatched lengths.")
        raise ValueError("Predictions and actual values have mismatched lengths.")

    X = np.column_stack((lstm_preds, svr_preds, xgboost_preds))
    y = actual_values

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained ensemble model
    try:
        joblib.dump(model, f'models/Ensemble/{company_id}.pkl')
        logger.info(f"Ensemble model for company_id {company_id} saved successfully.")
    except Exception as e:
        logger.error(f"Error saving the ensemble model: {e}")
        raise


if __name__ == "__main__":
    company_id = 1  # Example company_id
    try:
        train_ensemble(company_id)
    except Exception as e:
        logger.error(f"An error occurred: {e}")

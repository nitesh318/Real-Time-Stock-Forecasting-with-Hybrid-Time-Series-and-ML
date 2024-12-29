import logging
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}.")
        return data['Close'].values.reshape(-1, 1)
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def train_svr(data, company_id):
    try:
        # Scaling the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Prepare features and targets (predicting the next day's price)
        X, y = scaled_data[:-1], scaled_data[1:]

        # Time-series specific splitting (no random split)
        X_train, X_test, y_train, y_test = X[:-30], X[-30:], y[:-30], y[-30:]  # last 30 as test set

        # Initialize and train the SVR model
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(X_train, y_train.ravel())

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"SVR model for company {company_id} evaluation MSE: {mse:.4f}")

        # Ensure directories exist for saving model
        os.makedirs(f'models/SVR', exist_ok=True)

        # Save the trained model and scaler
        joblib.dump(model, f'models/SVR/{company_id}.pkl')
        joblib.dump(scaler, f'models/Scalers/{company_id}.pkl')
        logger.info(f"SVR model and scaler for company {company_id} saved successfully.")

    except Exception as e:
        logger.error(f"Error during SVR training for company {company_id}: {e}")
        raise


if __name__ == "__main__":
    company_id = 1  # Replace with the appropriate company ID
    file_path = 'data/stock_prices.csv'  # Replace with the correct file path

    # Load data and train the model
    data = load_data(file_path)
    train_svr(data, company_id)

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

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


def prepare_data(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


# Train LSTM
def train_lstm(data, company_id, look_back=30, epochs=10, batch_size=32):
    try:
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Prepare data
        X, y = prepare_data(scaled_data, look_back)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Define the LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, y, epochs=epochs, batch_size=batch_size)

        # Evaluate the model (optional)
        loss = model.evaluate(X, y)
        logger.info(f"Training completed. Model loss: {loss:.4f}")

        # Ensure the directories exist
        os.makedirs(f'models/LSTM', exist_ok=True)
        os.makedirs(f'models/Scalers', exist_ok=True)

        # Save the trained model and scaler
        model.save(f'models/LSTM/{company_id}.h5')
        joblib.dump(scaler, f'models/Scalers/{company_id}.pkl')
        logger.info(f"LSTM model and scaler for {company_id} saved successfully.")

    except Exception as e:
        logger.error(f"Error during LSTM training for company {company_id}: {e}")
        raise


if __name__ == "__main__":
    company_id = 1  # Replace with the appropriate company_id
    file_path = 'data/stock_prices.csv'  # Replace with the correct file path

    # Load data and train the model
    data = load_data(file_path)
    train_lstm(data, company_id)

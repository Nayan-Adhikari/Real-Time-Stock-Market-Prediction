import pandas as pd
import joblib
from fetch_data import fetch_real_time_data
from preprocess_data import preprocess_real_time_data

# File paths
MODEL_FILE = "stock_price_model.pkl"
REAL_TIME_DATA_FILE = "data/AAPL_real_time_data.csv"
REAL_TIME_CLEANED_DATA_FILE = "data/AAPL_real_time_data_cleaned.csv"

# Load the pre-trained model
model = joblib.load(MODEL_FILE)
print("Model loaded successfully.")

# Fetch real-time data for AAPL (or any other stock symbol)
real_time_data = fetch_real_time_data("AAPL")  # Make sure this function fetches real-time data correctly

# Preprocess the real-time data
real_time_data_cleaned = preprocess_real_time_data(REAL_TIME_DATA_FILE, REAL_TIME_CLEANED_DATA_FILE)

if real_time_data_cleaned is not None:
    # Select features for prediction (same as in training phase)
    features = ['Open', 'High', 'Low', 'Volume', 'MA_10']
    
    # Use the model to predict the closing price
    X_real_time = real_time_data_cleaned[features]
    predicted_close = model.predict(X_real_time)

    # Print predicted closing price
    print(f"Predicted closing price for AAPL: ${predicted_close[-1]:.2f}")

    # Save the predictions to a CSV file
    predictions = real_time_data_cleaned.copy()
    predictions['Predicted_Close'] = predicted_close
    predictions.to_csv("data/AAPL_real_time_predictions.csv", index=False)
    print("Predictions saved to AAPL_real_time_predictions.csv")
else:
    print("Error: Real-time data preprocessing failed.")

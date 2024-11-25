import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Load preprocessed data
DATA_DIR = "data"
file_path = f"{DATA_DIR}/AAPL_historical_data_cleaned.csv"
data = pd.read_csv(file_path)

# Define features and target
features = ['Open', 'High', 'Low', 'Volume', 'MA_10']  # Example features
target = 'Close'

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, "stock_price_model.pkl")
print("Model trained and saved to stock_price_model.pkl")

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

import yfinance as yf
import pandas as pd
import os

# Directory to store data
DATA_DIR = "data"

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def fetch_historical_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker symbol.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start_date (str): Start date in the format "YYYY-MM-DD".
        end_date (str): End date in the format "YYYY-MM-DD".

    Returns:
        DataFrame: Historical stock data.
    """
    print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Save data to CSV in the data directory
    file_path = os.path.join(DATA_DIR, f"{ticker}_historical_data.csv")
    data.to_csv(file_path, index=True)
    print(f"Data saved to {file_path}")

    return data


def fetch_real_time_data(ticker):
    """
    Fetch real-time stock data for the current day with 1-minute intervals.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL").

    Returns:
        DataFrame: Real-time stock data for the current day.
    """
    print(f"Fetching real-time data for {ticker}...")
    ticker_obj = yf.Ticker(ticker)

    # Get 1-minute interval data for the current day
    live_data = ticker_obj.history(period="1d", interval="1m")
    
    # Save data to CSV in the data directory
    file_path = os.path.join(DATA_DIR, f"{ticker}_real_time_data.csv")
    live_data.to_csv(file_path, index=True)
    print(f"Real-time data saved to {file_path}")

    return live_data


# Example Usage
if __name__ == "__main__":
    # Fetch historical data
    historical_data = fetch_historical_data("AAPL", "2020-01-01", "2024-11-20")
    print(historical_data.head())

    # Fetch real-time data
    real_time_data = fetch_real_time_data("AAPL")
    print(real_time_data.tail())

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from prometheus_client import start_http_server, Gauge
import time

# Load the pre-trained LSTM model
model = load_model('stock_price_lstm_model.h5')

# Initialize the MinMaxScaler (use the same scaler used for training)
scaler = MinMaxScaler(feature_range=(0, 1))

# Prometheus metrics
predicted_price_metric = Gauge('predicted_stock_price', 'Predicted stock price', ['symbol'])
actual_price_metric = Gauge('actual_stock_price', 'Actual stock price', ['symbol'])

# Start Prometheus server
start_http_server(8000)
print("Prometheus metrics server running on http://localhost:8000/metrics")

# Function to fetch the last N days of stock data
def fetch_live_data(symbol, period="7d", interval="1m"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    return df['Close'].values

# List of stock symbols to monitor
stock_symbols = ["AAPL", "GOOGL", "AMZN", "NVDA", "AIGFF"]  # Add your desired stock symbols

while True:
    for stock_symbol in stock_symbols:
        try:
            # Fetch the last 50 minutes of stock data
            live_data = fetch_live_data(stock_symbol, period="7d", interval="1m")[-50:]

            # Check if we have enough data
            if len(live_data) < 50:
                print(f"Not enough live data for {stock_symbol} yet. Waiting...")
                time.sleep(10)
                continue

            # Scale the data
            scaled_data = scaler.fit_transform(live_data.reshape(-1, 1))

            # Reshape for LSTM input
            input_data = np.array([scaled_data])
            
            # Make prediction
            predicted_price_scaled = model.predict(input_data)
            predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

            # Fetch the actual current price
            current_price = live_data[-1]

            # Update Prometheus metrics
            predicted_price_metric.labels(symbol=stock_symbol).set(predicted_price)
            actual_price_metric.labels(symbol=stock_symbol).set(current_price)

            # Print for debugging
            print(f"Predicted: {predicted_price:.2f} for {stock_symbol}, Actual: {current_price:.2f}")

        except Exception as e:
            print(f"Error fetching live data or making predictions for {stock_symbol}: {e}")
            time.sleep(10)

    # Wait for the next update
    time.sleep(10)  # Fetch data every 30 seconds

import yfinance as yf
import pandas as pd
from datetime import datetime

# --- Configuration ---
ticker_symbol = "^GSPC"
start_date = datetime(2024, 1, 1)  # Year, Month, Day
end_date = datetime(2025, 1, 1)    # Data will be downloaded up to, but not including, this date
output_filename = f"{ticker_symbol}_historical_data.csv"

# --- Data Download ---
print(f"Downloading historical data for {ticker_symbol} from {start_date.date()} to {end_date.date()}...")

try:
    # Use the yf.download function to fetch the data
    # interval='1d' means daily data (default if not specified)
    data = yf.download(
        tickers=ticker_symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True, # Automatically adjust Open, High, Low, Close for splits & dividends
        progress=False    # Set to True to see a download progress bar
    )

    # --- Data Processing and Saving ---
    if not data.empty:
        # Display the first few rows of the data
        print("\n--- First 5 Rows of Downloaded Data ---")
        print(data.head())
        
        # Save the data to a CSV file
        data.to_csv(output_filename)
        print(f"\nSuccessfully downloaded and saved data to {output_filename}")
    else:
        print(f"\nNo data found for the ticker {ticker_symbol} in the specified date range.")

except Exception as e:
    print(f"\nAn error occurred during download: {e}")
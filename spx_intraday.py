import yfinance as yf
import pandas as pd

# --- Configuration ---
ticker_symbol = "SPY"   # S&P 500 ETF (trades like a stock, shorter after-hours)
intraday_period = "60d"  # Specifies the last 5 days of data
intraday_interval = "5m" # Specifies 1-minute data granularity
output_filename = f"{ticker_symbol}_intraday_{intraday_interval}_{intraday_period}.csv"

# --- Data Download ---
print(f"Downloading {intraday_interval} data for {ticker_symbol} over the last {intraday_period}...")

try:
    # Use the yf.download function
    # Note: SPY often includes limited pre-market (4:00 AM - 9:30 AM ET) 
    # and after-hours (4:00 PM - 8:00 PM ET) data, but this is less extensive 
    # than the futures data (^GSPC).
    data = yf.download(
        tickers=ticker_symbol,
        period=intraday_period,
        interval=intraday_interval,
        auto_adjust=True,  
        progress=False
    )

    # --- Data Processing and Saving ---
    if not data.empty:
        # Display the first few rows of the data to check the new timestamps
        print("\n--- First 5 Rows of Downloaded SPY Intraday Data (Check Time Zone / Hours) ---")
        print(data.head())
        
        # Save the data to a CSV file
        data.to_csv(output_filename)
        print(f"\nSuccessfully downloaded and saved {len(data)} data points to {output_filename}")
        
        # Example of how to filter the data to ONLY the regular market hours (9:30 AM to 4:00 PM ET/EST)
        # Note: Timezone handling in Python/Pandas can be complex, but this uses simple time filtering 
        # on the UTC index assuming yfinance returns a consistent index format.
        print("\n--- Example: Filtering Data to Regular Trading Hours (9:30 AM - 4:00 PM ET) ---")
        
        # 9:30 AM ET is 14:30 UTC
        # 4:00 PM ET is 21:00 UTC
        
        # Ensure the index is a timezone-aware DatetimeIndex
        if data.index.tz is None:
            # We assume yfinance often returns it in UTC without the label, so let's localize it and convert
            # This step is often necessary for reliable time-based filtering
            data = data.tz_localize('UTC').tz_convert('America/New_York')

        # Filter the data to only include times between 9:30 AM and 4:00 PM New York Time
        regular_hours_data = data.between_time('09:30', '16:00', include_start=True, include_end=False)
        
        print(f"Filtered to {len(regular_hours_data)} data points in regular hours.")
        print(regular_hours_data.head())


    else:
        print(f"\nNo data found for the ticker {ticker_symbol} with the specified period/interval.")

except Exception as e:
    print(f"\nAn error occurred during download: {e}")
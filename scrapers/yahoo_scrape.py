import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import timedelta

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "capitol trades new" / "capitol_trades_clean.csv"
OUTPUT_PATH = BASE_DIR / "yahoo_finance_data.csv"

def get_ticker_data(tickers, start_date, end_date):
    """
    Fetches historical stock data for a list of tickers from Yahoo Finance.
    """
    if not tickers:
        return pd.DataFrame()
        
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

def main():
    """
    Main function to fetch and cache Yahoo Finance data efficiently.
    - Fetches full history for new tickers.
    - Fetches only recent, missing data for existing tickers.
    """
    if not INPUT_PATH.exists():
        print(f"Error: Input file not found at {INPUT_PATH}")
        return

    capitol_trades_df = pd.read_csv(INPUT_PATH)
    capitol_trades_df['traded_dt'] = pd.to_datetime(capitol_trades_df['traded_dt'], errors='coerce')
    capitol_trades_df.dropna(subset=['traded_dt'], inplace=True)

    if capitol_trades_df.empty:
        print("No valid trade dates found. Exiting.")
        return
        
    # --- Determine required date range from all trades ---
    required_start_date = capitol_trades_df['traded_dt'].min()
    required_end_date = capitol_trades_df['traded_dt'].max() + timedelta(days=1)
    
    required_tickers = capitol_trades_df['ticker'].unique().tolist()
    
    # --- Load Cache and Identify New vs. Existing Tickers ---
    cached_df = pd.DataFrame()
    if OUTPUT_PATH.exists():
        print(f"Loading cached Yahoo Finance data from {OUTPUT_PATH}...")
        cached_df = pd.read_csv(OUTPUT_PATH, header=[0, 1], index_col=0)
        cached_df.index = pd.to_datetime(cached_df.index) # Ensure index is datetime
        existing_tickers = cached_df.columns.get_level_values(0).unique().tolist()
        existing_tickers = [t for t in existing_tickers if 'Unnamed' not in t]
        print(f"Found {len(existing_tickers)} existing tickers in cache.")
    else:
        existing_tickers = []
        print("No cache file found.")

    new_tickers = [t for t in required_tickers if t not in existing_tickers]

    # --- Fetch Full History for New Tickers ---
    new_data = get_ticker_data(
        new_tickers, 
        required_start_date.strftime('%Y-%m-%d'), 
        required_end_date.strftime('%Y-%m-%d')
    )

    # --- Fetch Incremental Updates for Existing Tickers ---
    updated_data_for_existing_tickers = pd.DataFrame()
    if not cached_df.empty:
        cache_end_date = cached_df.index.max()
        if required_end_date > cache_end_date:
            print("Cache is outdated. Fetching incremental data for existing tickers...")
            update_start_date = cache_end_date + timedelta(days=1)
            updated_data_for_existing_tickers = get_ticker_data(
                existing_tickers,
                update_start_date.strftime('%Y-%m-%d'),
                required_end_date.strftime('%Y-%m-%d')
            )
            # Append new data to the cached data
            cached_df = pd.concat([cached_df, updated_data_for_existing_tickers])

    # --- Combine and Save ---
    # Combine the updated cache with the data for brand new tickers
    final_df = pd.concat([cached_df, new_data], axis=1)
    
    if final_df.empty:
        print("No data to save.")
        return

    final_df.sort_index(axis=1, inplace=True) # Sort columns alphabetically by ticker
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH)
    print(f"Updated Yahoo Finance data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

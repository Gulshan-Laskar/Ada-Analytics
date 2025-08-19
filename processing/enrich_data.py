import pandas as pd
import yfinance as yf
from pathlib import Path
import time
import re
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "capitol trades new" / "capitol_trades_clean.csv"
OUTPUT_PATH = BASE_DIR / "data" / "enriched_trades.csv"

def clean_ticker_symbol(ticker):
    """
    Cleans common non-standard ticker formats and filters out invalid ones.
    """
    if pd.isna(ticker):
        return None
    ticker = str(ticker).upper().strip()
    
    # More aggressive cleaning
    prefixes_to_remove = ['PLC', 'INC', 'CORP', 'ETF', 'NV', 'SA', 'AG', 'LTD', 'ASA', 'SE', 'LP']
    pattern = r'^(' + '|'.join(prefixes_to_remove) + r')'
    ticker = re.sub(pattern, '', ticker)
    ticker = re.sub(r'^[.\s]+|[.\sW]+$', '', ticker) # Remove leading/trailing junk and trailing 'W'

    # Filter out clearly invalid tickers to avoid API calls
    # Allow single-letter tickers as they can be valid (e.g., F, T)
    if len(ticker) < 1 or not re.match(r'^[A-Z0-9.-]+$', ticker):
        return None
        
    return ticker

def fetch_enrichment_data(tickers_to_fetch):
    """
    Fetches market cap and sector ONLY for a list of new tickers.
    Uses tqdm for a progress bar.
    """
    if not tickers_to_fetch:
        return {}

    print(f"Fetching new data for {len(tickers_to_fetch)} tickers...")
    enrichment_data = {}
    for ticker in tqdm(tickers_to_fetch, desc="Enriching Tickers"):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap')
            sector = info.get('sector')
            
            if market_cap is None:
                print(f"\nWarning: Could not find market cap for {ticker}.")

            enrichment_data[ticker] = {'market_cap': market_cap, 'sector': sector}
            time.sleep(0.25) # Be respectful to the API
        except Exception:
            # If a ticker fails, we still record it as processed to avoid re-fetching
            print(f"\nWarning: Could not fetch data for '{ticker}'. It may be delisted or invalid.")
            enrichment_data[ticker] = {'market_cap': None, 'sector': 'Fetch Error'}
            
    return enrichment_data

def main():
    """
    Enriches data, using the output file as a persistent cache to only fetch new tickers.
    """
    print("--- Starting Step 1: Data Enrichment ---")
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found at {INPUT_PATH}. Please run cleaning script first.")
        return

    df = pd.read_csv(INPUT_PATH)
    
    # --- Clean ticker symbols ---
    df['ticker_original'] = df['ticker']
    df['ticker'] = df['ticker_original'].apply(clean_ticker_symbol)

    # --- Persistent Caching Logic ---
    # Use the existing output file as a cache to avoid re-fetching everything
    processed_tickers_cache = {}
    if OUTPUT_PATH.exists():
        print(f"Loading existing cache from {OUTPUT_PATH}...")
        cached_df = pd.read_csv(OUTPUT_PATH)
        # Create a lookup dictionary from the cache for already processed tickers
        cached_df.dropna(subset=['ticker'], inplace=True)
        processed_tickers_cache = cached_df.set_index('ticker')[['market_cap', 'sector']].to_dict('index')
        print(f"Found {len(processed_tickers_cache)} tickers in cache.")

    # Determine which tickers are new and need to be fetched
    all_unique_tickers = df['ticker'].dropna().unique()
    tickers_to_fetch = [t for t in all_unique_tickers if t not in processed_tickers_cache]

    # Fetch data ONLY for the new tickers
    new_enrichment_data = fetch_enrichment_data(tickers_to_fetch)
    
    # Combine new data with the cache to create a full enrichment map
    full_enrichment_map = {**processed_tickers_cache, **new_enrichment_data}

    # --- Map enriched data back to the main dataframe ---
    df['market_cap'] = df['ticker'].map(lambda t: full_enrichment_map.get(t, {}).get('market_cap'))
    df['sector'] = df['ticker'].map(lambda t: full_enrichment_map.get(t, {}).get('sector'))
    print("Enriched data with market cap and sector.")

    # --- Final Calculations ---
    df['size_avg_usd'] = df[['size_low_usd', 'size_high_usd']].mean(axis=1)
    df['trade_size_vs_market_cap'] = df.apply(
        lambda row: (row['size_avg_usd'] / row['market_cap']) * 100 
                    if pd.notnull(row['market_cap']) and row['market_cap'] > 0 
                    else None,
        axis=1
    )
    print("Calculated trade size and normalized by market cap.")
    
    # --- Save Output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"--- Enrichment complete. Data saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()

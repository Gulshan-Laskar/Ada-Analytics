import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "master_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "sell_signals.csv"

def main():
    """
    Identifies and scores potential SELL signals from the master dataset.
    This script focuses only on 'sale' or 'sell' type transactions.
    """
    print("--- Generating Sell Signals ---")
    if not INPUT_FILE.exists():
        print(f"ERROR: Master dataset not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, parse_dates=["published_dt"])

    # --- Filter for Sell-type Transactions ---
    # We are only interested in when politicians are selling
    sell_df = df[df['type'].str.contains('sale', case=False, na=False)].copy()

    if sell_df.empty:
        print("No sell transactions found in the master dataset.")
        # Create an empty file to signify a completed run
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False)
        return

    # --- Ensure numeric types for feature calculations ---
    feature_cols = ["size_avg_usd", "trade_size_vs_market_cap", "sentiment_score",
                    "mention_count", "volume_spike_ratio", "consensus_score_7d"]
    for col in feature_cols:
        if col in sell_df.columns:
            sell_df[col] = pd.to_numeric(sell_df[col], errors='coerce').fillna(0)

    # --- Define Sell Signal Logic ---
    # A strong sell signal might be a large trade with high consensus (many politicians selling)
    big_trade = (sell_df["trade_size_vs_market_cap"] >= sell_df["trade_size_vs_market_cap"].quantile(0.8)) | \
                (sell_df["size_avg_usd"] >= sell_df["size_avg_usd"].quantile(0.8))
    
    consensus_sell = sell_df["consensus_score_7d"] >= 1  # At least one other politician sold recently

    # A spike in negative sentiment could also be a strong signal
    negative_sentiment_spike = sell_df["sentiment_score"] <= sell_df["sentiment_score"].quantile(0.2)

    # Combine the logic to create the final signal
    signal = big_trade & (consensus_sell | negative_sentiment_spike)

    # Assign a signal strength (0 or 1)
    sell_df["sell_signal_strength"] = np.where(signal, 1, 0)
    
    # --- Save the Output ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sell_df.to_csv(OUTPUT_FILE, index=False)

    print(f"--- Sell signals generated successfully. Output saved to {OUTPUT_FILE} ---")
    print(f"Found {len(sell_df[sell_df['sell_signal_strength'] == 1])} high-confidence sell signals.")

if __name__ == "__main__":
    main()

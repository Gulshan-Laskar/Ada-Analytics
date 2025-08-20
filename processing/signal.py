import pandas as pd
import numpy as np

# ==== EDIT THESE PATHS ====
INPUT_FILE = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/master_dataset.csv"
OUTPUT_FILE = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/signals.csv"
# ==========================

# Load dataset
df = pd.read_csv(INPUT_FILE)

# --- Parse dates ---
df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")

# --- Fill missing tickers from ticker_original ---
df["ticker"] = df["ticker"].fillna(df["ticker_original"])
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

# --- Ensure numeric ---
for col in ["size_avg_usd", "trade_size_vs_market_cap", "sentiment_score",
            "mention_count", "volume_spike_ratio", "consensus_score_7d"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# --- Simple thresholds ---
recent_disclosure = (df["published_dt"].max() - df["published_dt"]).dt.days <= 14
big_trade = (df["trade_size_vs_market_cap"] >= df["trade_size_vs_market_cap"].quantile(0.9)) | \
            (df["size_avg_usd"] >= df["size_avg_usd"].quantile(0.9))
sentiment_spike = df["sentiment_score"] >= df["sentiment_score"].mean() + df["sentiment_score"].std()
mention_spike   = df["mention_count"] >= df["mention_count"].mean() + df["mention_count"].std()
volume_spike    = df["volume_spike_ratio"] >= df["volume_spike_ratio"].quantile(0.75)
consensus_pos   = df["consensus_score_7d"] >= 0.5

# --- Define binary signal ---
signal = recent_disclosure & big_trade & (
    (sentiment_spike & (mention_spike | volume_spike)) |
    (consensus_pos & (mention_spike | volume_spike))
)

# --- Assign signal strength (0 or 1) ---
df["signal_strength"] = np.where(signal, 1, 0)

# --- Save ALL rows ---
df.to_csv(OUTPUT_FILE, index=False)


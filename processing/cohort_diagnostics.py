import pandas as pd
import numpy as np
from pathlib import Path

# ==== EDIT PATHS ====
TRADES_FILE = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/trades_backtest.csv"   # from your backtest
SIGNALS_FILE = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/signals.csv"          # from your signal generator
OUT_DIR = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/cohort_diagnostics"            # outputs will be written here
# =====================

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# after reading the CSVs
tr = pd.read_csv(TRADES_FILE, parse_dates=["published_dt","entry_date","exit_date"])
sg = pd.read_csv(SIGNALS_FILE)

# --- FIX: align datatypes ---
tr["published_dt"] = pd.to_datetime(tr["published_dt"], errors="coerce")
sg["published_dt"] = pd.to_datetime(sg["published_dt"], errors="coerce")

# continue with your merge
df = tr.merge(sg, on=["ticker","published_dt"], how="left")


# Join key signal columns to trades for grouping
cols = [
    "ticker","published_dt","company","politician_name","type","owner","sector",
    "size_avg_usd","trade_size_vs_market_cap","sentiment_score","mention_count",
    "volume_spike_ratio","consensus_score_7d","signal_strength"
]
sg = sg[[c for c in cols if c in sg.columns]].copy()
df = tr.merge(sg, on=["ticker","published_dt"], how="left")

def summarize(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    wr = float((g["net_return"] > 0).mean()) if n else np.nan
    avg = float(g["net_return"].mean()) if n else np.nan
    med = float(g["net_return"].median()) if n else np.nan
    std = float(g["net_return"].std(ddof=1)) if n else np.nan
    sharpe = (avg / std) if (std and std > 0) else np.nan
    return pd.Series({"trades": n, "win_rate": wr, "avg": avg, "median": med, "sharpe": sharpe})

# by ticker
df.groupby("ticker").apply(summarize).sort_values(["sharpe","trades"], ascending=[False,False]) \
  .to_csv(f"{OUT_DIR}/by_ticker.csv")

# by politician
if "politician_name" in df.columns:
    df.groupby("politician_name").apply(summarize).sort_values(["sharpe","trades"], ascending=[False,False]) \
      .to_csv(f"{OUT_DIR}/by_politician.csv")

# by sector
if "sector" in df.columns:
    df.groupby("sector").apply(summarize).sort_values(["sharpe","trades"], ascending=[False,False]) \
      .to_csv(f"{OUT_DIR}/by_sector.csv")

# by deciles/bins
def qbin(s, q=10):
    return pd.qcut(s.fillna(0), q, labels=False, duplicates="drop")

if "trade_size_vs_market_cap" in df.columns:
    df["tsvm_decile"] = qbin(df["trade_size_vs_market_cap"], 10)
    df.groupby("tsvm_decile").apply(summarize).sort_index() \
      .to_csv(f"{OUT_DIR}/by_tsvm_decile.csv")

if "sentiment_score" in df.columns:
    df["sent_z_bin"] = qbin(df["sentiment_score"], 10)
    df.groupby("sent_z_bin").apply(summarize).sort_index() \
      .to_csv(f"{OUT_DIR}/by_sentiment_bin.csv")

if "mention_count" in df.columns:
    df["mentions_bin"] = qbin(df["mention_count"], 10)
    df.groupby("mentions_bin").apply(summarize).sort_index() \
      .to_csv(f"{OUT_DIR}/by_mentions_bin.csv")

if "volume_spike_ratio" in df.columns:
    df["vol_spike_bin"] = qbin(df["volume_spike_ratio"], 10)
    df.groupby("vol_spike_bin").apply(summarize).sort_index() \
      .to_csv(f"{OUT_DIR}/by_volume_spike_bin.csv")


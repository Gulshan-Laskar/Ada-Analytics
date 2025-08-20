import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
TRADES_FILE = BASE_DIR / "data" / "trades_backtest.csv"
SIGNALS_FILE = BASE_DIR / "data" / "signals.csv"
OUT_DIR = BASE_DIR / "data" / "cohort_diagnostics"

def summarize(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    if n == 0: return pd.Series({"trades": 0, "win_rate": np.nan, "avg": np.nan, "median": np.nan, "sharpe": np.nan})
    wr = (g["net_return"] > 0).mean()
    avg = g["net_return"].mean()
    med = g["net_return"].median()
    std = g["net_return"].std(ddof=1)
    sharpe = avg / std if std > 0 else np.nan
    return pd.Series({"trades": n, "win_rate": wr, "avg": avg, "median": med, "sharpe": sharpe})

def qbin(s, q=10):
    return pd.qcut(s.fillna(0), q, labels=False, duplicates="drop")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(TRADES_FILE, parse_dates=["published_dt","entry_date","exit_date"])
    sg = pd.read_csv(SIGNALS_FILE, parse_dates=["published_dt"])

    # Merge trades with original signals to get all feature columns for analysis
    df = tr.merge(sg, on=["ticker","published_dt"], how="left")

    # --- Run diagnostics by various cohorts ---
    cohorts = {
        "ticker": "by_ticker.csv",
        "politician_name": "by_politician.csv",
        "sector": "by_sector.csv"
    }
    for col, fname in cohorts.items():
        if col in df.columns:
            df.groupby(col).apply(summarize).sort_values(["sharpe","trades"], ascending=False).to_csv(OUT_DIR / fname)
            print(f"[OK] Wrote cohort analysis: {OUT_DIR / fname}")

    # --- Run diagnostics by feature bins ---
    binned_features = {
        "trade_size_vs_market_cap": "by_tsvm_decile.csv",
        "sentiment_score": "by_sentiment_bin.csv",
        "mention_count": "by_mentions_bin.csv",
        "volume_spike_ratio": "by_volume_spike_bin.csv"
    }
    for col, fname in binned_features.items():
        if col in df.columns:
            bin_col_name = f"{col}_bin"
            df[bin_col_name] = qbin(df[col])
            df.groupby(bin_col_name).apply(summarize).sort_index().to_csv(OUT_DIR / fname)
            print(f"[OK] Wrote binned analysis: {OUT_DIR / fname}")

if __name__ == "__main__":
    main()

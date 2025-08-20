import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import pickle

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
SIGNALS_FILE = BASE_DIR / "data" / "signals.csv"
TRADES_OUT = BASE_DIR / "data" / "trades_backtest.csv"
SUMMARY_OUT = BASE_DIR / "data" / "backtest_summary.csv"
PRICE_CACHE_FILE = BASE_DIR / "data" / "price_cache.pkl"
HOLDING_DAYS = 5
COST_BPS = 10

def load_price_cache():
    if PRICE_CACHE_FILE.exists():
        with open(PRICE_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_price_cache(cache):
    with open(PRICE_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def update_price_cache(tickers, start, end, cache):
    tickers_to_fetch = [t for t in tickers if t not in cache]
    if not tickers_to_fetch:
        print("All required prices found in cache.")
        return cache

    print(f"Fetching prices for {len(tickers_to_fetch)} new tickers...")
    data = yf.download(tickers_to_fetch, start=start, end=end, progress=False, auto_adjust=False, actions=False, interval="1d", group_by='ticker')
    
    for t in tickers_to_fetch:
        try:
            hist = data[t] if len(tickers_to_fetch) > 1 else data
            if not hist.empty:
                hist = hist[["Open","Close"]].dropna()
                hist.index = pd.to_datetime(hist.index)
                cache[t] = hist.groupby(level=0).first().sort_index()
        except KeyError:
            print(f"[WARN] No data returned for {t}")
    return cache

def load_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df.dropna(subset=["published_dt", "ticker"], inplace=True)
    return df.drop_duplicates(subset=["ticker", "published_dt"]).reset_index(drop=True)

def next_trading_day(prices: pd.DataFrame, after_date: pd.Timestamp) -> pd.Timestamp | None:
    pos = prices.index.searchsorted(after_date + pd.Timedelta(days=1))
    return prices.index[pos] if pos < len(prices.index) else None

def _scalar(v) -> float | None:
    if isinstance(v, pd.Series): v = v.iloc[0] if len(v) else np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except Exception: return None

def simulate_trade_time_exit(prices: pd.DataFrame, signal_date: pd.Timestamp) -> dict:
    if prices is None or prices.empty: return {"skipped": True, "reason": "no_prices"}
    entry_date = next_trading_day(prices, signal_date)
    if entry_date is None: return {"skipped": True, "reason": "no_entry"}
    entry_open = _scalar(prices.get("Open", {}).get(entry_date))
    if entry_open is None: return {"skipped": True, "reason": "bad_entry_price"}
    
    start_pos = prices.index.get_loc(entry_date)
    exit_pos = min(start_pos + HOLDING_DAYS - 1, len(prices.index) - 1)
    exit_date = prices.index[exit_pos]
    exit_close = _scalar(prices.get("Close", {}).get(exit_date))
    if exit_close is None: return {"skipped": True, "reason": "bad_exit_price"}

    gross_ret = (exit_close - entry_open) / entry_open
    return {
        "skipped": False, "entry_date": entry_date, "entry_price": entry_open,
        "exit_date": exit_date, "exit_price": exit_close, "gross_return": gross_ret,
        "net_return": gross_ret - (COST_BPS / 10000.0), "exit_reason": "time_exit",
        "holding_days": (exit_date - entry_date).days + 1
    }

def main():
    signals = load_signals(SIGNALS_FILE)
    if signals.empty:
        print("[ERROR] No signals found."); return

    min_dt, max_dt = signals["published_dt"].min(), signals["published_dt"].max()
    
    price_cache = load_price_cache()
    price_cache = update_price_cache(sorted(signals["ticker"].unique()), min_dt - pd.Timedelta(days=15), max_dt + pd.Timedelta(days=15), price_cache)
    save_price_cache(price_cache)

    rows = [
        {"ticker": r["ticker"], "published_dt": r["published_dt"], **simulate_trade_time_exit(price_cache.get(r["ticker"]), r["published_dt"])}
        for _, r in signals.iterrows()
    ]
    
    executed = pd.DataFrame(rows)
    executed = executed[~executed["skipped"]].copy().sort_values(["ticker","entry_date"])
    TRADES_OUT.parent.mkdir(parents=True, exist_ok=True)
    executed.to_csv(TRADES_OUT, index=False)

    n = len(executed)
    if n > 0:
        win_rate = (executed["net_return"] > 0).mean()
        avg_ret = executed["net_return"].mean()
        sharpe = avg_ret / executed["net_return"].std(ddof=1) if executed["net_return"].std(ddof=1) > 0 else np.nan
    else:
        win_rate = avg_ret = sharpe = np.nan

    summary = pd.DataFrame([{"trades": n, "win_rate": win_rate, "avg_trade_return": avg_ret, "sharpe_per_trade": sharpe, "holding_days": HOLDING_DAYS, "cost_bps": COST_BPS}])
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"[OK] Saved trades to: {TRADES_OUT}")
    print(f"[OK] Saved summary to: {SUMMARY_OUT}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

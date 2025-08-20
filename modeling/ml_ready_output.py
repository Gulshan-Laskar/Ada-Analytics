import re
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from collections import Counter
import pickle

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
SIGNALS_FILE = BASE_DIR / "data" / "signals.csv"
ML_OUT = BASE_DIR / "data" / "ml_dataset.csv"
SKIPPED_OUT = BASE_DIR / "data" / "ml_skipped_symbols.xlsx"
PRICE_CACHE_FILE = BASE_DIR / "data" / "price_cache.pkl"

VALID_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\-\.]{0,9}$")

def load_price_cache():
    if PRICE_CACHE_FILE.exists():
        with open(PRICE_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_price_cache(cache):
    with open(PRICE_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def normalize_ticker(t: str) -> str:
    if not isinstance(t, str): return ""
    return re.sub(r"\s+", "", t.upper().strip().replace(".", "-"))

def looks_valid(t: str) -> bool:
    if not t or len(t) > 10: return False
    return bool(VALID_TICKER_RE.match(t))

def update_price_cache(tickers, start, end, cache):
    tickers_to_fetch = [t for t in tickers if t not in cache]
    if not tickers_to_fetch:
        print("All required prices found in cache.")
        return cache

    print(f"Fetching prices for {len(tickers_to_fetch)} new tickers...")
    data = yf.download(tickers_to_fetch, start=start, end=end, progress=False, auto_adjust=False, actions=False, interval="1d", group_by='ticker')
    
    for t in tickers_to_fetch:
        try:
            # Handle single vs multi ticker download format
            if len(tickers_to_fetch) == 1:
                hist = data
            else:
                hist = data[t]
            
            if not hist.empty:
                hist = hist[["Open","Close"]].dropna()
                hist.index = pd.to_datetime(hist.index)
                cache[t] = hist.groupby(level=0).first().sort_index()
        except KeyError:
            print(f"[WARN] No data returned for {t}")
    return cache

def next_trading_day(idx: pd.DatetimeIndex, after: pd.Timestamp):
    pos = idx.searchsorted(after + pd.Timedelta(days=1))
    return idx[pos] if pos < len(idx) else None

def to_scalar(x):
    if isinstance(x, pd.Series): x = x.iloc[0] if len(x) else np.nan
    try:
        f = float(x)
        return f if np.isfinite(f) and f > 0 else None
    except Exception: return None

def forward_return_series(px: pd.DataFrame, published_dt: pd.Timestamp, horizons=(1,3,5,10)):
    out = {}
    if px is None or px.empty or pd.isna(published_dt): return out
    entry_date = next_trading_day(px.index, published_dt)
    if entry_date is None: return out
    e_open = to_scalar(px.get("Open", {}).get(entry_date))
    if e_open is None: return out
    
    pos0 = px.index.get_loc(entry_date)
    for h in horizons:
        pos1 = pos0 + h - 1
        if pos1 < len(px.index):
            d1 = px.index[pos1]
            c1 = to_scalar(px.get("Close", {}).get(d1))
            if c1 is not None:
                out[f"fwd_{h}d_ret"] = (c1 - e_open) / e_open
    return out

def main():
    sig = pd.read_csv(SIGNALS_FILE)
    sig["published_dt"] = pd.to_datetime(sig["published_dt"], errors="coerce")
    if 'ticker_original' in sig.columns:
        sig["ticker"] = sig["ticker"].fillna(sig["ticker_original"])
    sig["ticker"] = sig["ticker"].astype(str).map(normalize_ticker)

    valid_mask = sig["ticker"].map(looks_valid)
    sig_ok = sig[valid_mask & sig["published_dt"].notna()].drop_duplicates(subset=["ticker","published_dt"]).reset_index(drop=True)

    if sig_ok.empty:
        print("[ERROR] No valid signals after filtering."); return

    min_dt, max_dt = sig_ok["published_dt"].min(), sig_ok["published_dt"].max()
    
    price_cache = load_price_cache()
    price_cache = update_price_cache(sorted(sig_ok["ticker"].unique()), min_dt - pd.Timedelta(days=15), max_dt + pd.Timedelta(days=15), price_cache)
    save_price_cache(price_cache)

    rows, skipped_rows = [], []
    keep_cols = [c for c in ["published_dt","ticker","company","politician_name","type","owner","sector","size_avg_usd","trade_size_vs_market_cap","sentiment_score","mention_count","volume_spike_ratio","consensus_score_7d","signal_strength"] if c in sig_ok.columns]

    for _, r in sig_ok.iterrows():
        px = price_cache.get(r["ticker"])
        fwd = forward_return_series(px, r["published_dt"])
        if not fwd:
            skipped_rows.append({"ticker": r["ticker"], "published_dt": r["published_dt"], "reason": "no_forward_prices"})
            continue
        row = r[keep_cols].to_dict()
        row.update(fwd)
        rows.append(row)

    out = pd.DataFrame(rows)
    ML_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ML_OUT, index=False)

    pd.DataFrame(skipped_rows).to_excel(SKIPPED_OUT, sheet_name="no_forward_prices", index=False)

    print(f"[OK] Wrote ML dataset: {ML_OUT} with shape {out.shape}")
    print(f"[OK] Wrote skipped report: {SKIPPED_OUT}")

if __name__ == "__main__":
    main()

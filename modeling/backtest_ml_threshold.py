import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
SCORED_FILE = BASE_DIR / "data" / "scored_test.csv"
TRADES_OUT = BASE_DIR / "data" / "trades_backtest_ml.csv"
SUMMARY_OUT = BASE_DIR / "data" / "backtest_summary_ml.csv"
PROBA_THRESHOLD = 0.70
HOLDING_DAYS = 5
COST_BPS = 10

def _scalar(v):
    if isinstance(v, pd.Series): v = v.iloc[0] if len(v) else np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except Exception: return None

def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df.dropna(subset=["published_dt","ticker","proba_best"], inplace=True)
    return df[df["proba_best"] >= PROBA_THRESHOLD].drop_duplicates(subset=["ticker","published_dt"]).reset_index(drop=True)

def fetch_prices(tickers, start, end):
    start_buf, end_buf = start - pd.Timedelta(days=10), end + pd.Timedelta(days=10)
    data = {}
    for t in sorted(set(tickers)):
        try:
            hist = yf.download(t, start=start_buf, end=end_buf, progress=False, auto_adjust=False, actions=False, interval="1d")
            if not hist.empty:
                px = hist[["Open","Close"]].copy()
                px.index = pd.to_datetime(px.index).tz_localize(None)
                data[t] = px.groupby(level=0).first().sort_index()
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    return data

def next_trading_day(idx: pd.DatetimeIndex, after: pd.Timestamp):
    pos = idx.searchsorted(after + pd.Timedelta(days=1))
    return idx[pos] if pos < len(idx) else None

def simulate_time_exit(px: pd.DataFrame, published_dt: pd.Timestamp):
    if px is None or px.empty: return None
    entry_dt = next_trading_day(px.index, published_dt)
    if entry_dt is None: return None
    o = _scalar(px.loc[entry_dt, "Open"])
    if o is None: return None
    
    pos0 = px.index.get_loc(entry_dt)
    pos1 = min(pos0 + HOLDING_DAYS - 1, len(px.index) - 1)
    exit_dt = px.index[pos1]
    c = _scalar(px.loc[exit_dt, "Close"])
    if c is None: return None

    gross = (c - o) / o
    return {
        "entry_date": entry_dt, "entry_price": o, "exit_date": exit_dt, "exit_price": c,
        "gross_return": gross, "net_return": gross - (COST_BPS / 10000.0),
        "exit_reason": "time_exit", "holding_days": (exit_dt - entry_dt).days + 1
    }

def main():
    filt = load_scored(SCORED_FILE)
    if filt.empty:
        print(f"[ERROR] No rows pass threshold {PROBA_THRESHOLD}."); return

    min_dt, max_dt = filt["published_dt"].min(), filt["published_dt"].max()
    prices = fetch_prices(filt["ticker"].tolist(), min_dt, max_dt)

    rows = []
    for _, r in filt.iterrows():
        sim = simulate_time_exit(prices.get(r["ticker"]), r["published_dt"])
        if sim:
            rows.append({"ticker": r["ticker"], "published_dt": r["published_dt"], "proba_best": r["proba_best"], **sim})

    trades = pd.DataFrame(rows).sort_values(["published_dt","ticker"])
    TRADES_OUT.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES_OUT, index=False)

    n = len(trades)
    if n > 0:
        wr = (trades["net_return"] > 0).mean()
        avg = trades["net_return"].mean()
        sharpe = avg / trades["net_return"].std(ddof=1) if trades["net_return"].std(ddof=1) > 0 else np.nan
    else:
        wr = avg = sharpe = np.nan

    summary = pd.DataFrame([{"trades": n, "win_rate": wr, "avg_trade_return": avg, "sharpe_per_trade": sharpe, "holding_days": HOLDING_DAYS, "cost_bps": COST_BPS, "proba_threshold": PROBA_THRESHOLD}])
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"[OK] ML-filtered trades: {len(trades)} -> {TRADES_OUT}")
    print(f"[OK] Summary -> {SUMMARY_OUT}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

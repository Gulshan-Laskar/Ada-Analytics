# backtest_ml_threshold.py
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

SCORED_FILE  = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/scored_test.csv"
TRADES_OUT   = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/trades_backtest_ml.csv"
SUMMARY_OUT  = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/backtest_summary_ml.csv"

PROBA_THRESHOLD = 0.70
HOLDING_DAYS    = 5
COST_BPS        = 10

def _scalar(v):
    """Coerce a cell (scalar/Series) to a finite positive float or return None."""
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except Exception:
        return None

def load_scored(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["published_dt","ticker","proba_best"]:
        if col not in df.columns:
            raise ValueError(f"{path} must contain column '{col}'")
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["published_dt","ticker","proba_best"])
    df = df[df["proba_best"] >= PROBA_THRESHOLD]
    df = df.drop_duplicates(subset=["ticker","published_dt"]).reset_index(drop=True)
    return df

def fetch_prices(tickers, start, end):
    start_buf = start - pd.Timedelta(days=10)
    end_buf   = end + pd.Timedelta(days=10)
    data = {}
    for t in sorted(set(tickers)):
        try:
            hist = yf.download(
                t,
                start=start_buf.strftime("%Y-%m-%d"),
                end=end_buf.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=False,
                actions=False,
            )
            if not hist.empty:
                px = hist[["Open","Close"]].copy()
                px.index = pd.to_datetime(px.index, errors="coerce")
                # strip tz, drop NaT, collapse duplicate dates
                if getattr(px.index, "tz", None) is not None:
                    px.index = px.index.tz_localize(None)
                px = px[~px.index.isna()]
                px = px.groupby(level=0).first().sort_index()
                data[t] = px
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    return data

def next_trading_day(idx: pd.DatetimeIndex, after: pd.Timestamp):
    pos = idx.searchsorted(after + pd.Timedelta(days=1))
    return idx[pos] if pos < len(idx) else None

def simulate_time_exit(px: pd.DataFrame, published_dt: pd.Timestamp):
    if px is None or px.empty:
        return None
    entry_dt = next_trading_day(px.index, published_dt)
    if entry_dt is None or entry_dt not in px.index:
        return None

    o = _scalar(px.loc[entry_dt, "Open"])
    if o is None:
        return None

    pos0 = px.index.get_loc(entry_dt)
    pos1 = min(pos0 + HOLDING_DAYS - 1, len(px.index) - 1)
    exit_dt = px.index[pos1]

    c = _scalar(px.loc[exit_dt, "Close"])
    if c is None:
        return None

    gross = (c - o) / o
    net = gross - (COST_BPS / 10000.0)
    return {
        "entry_date": entry_dt,
        "entry_price": float(o),
        "exit_date": exit_dt,
        "exit_price": float(c),
        "gross_return": float(gross),
        "net_return": float(net),
        "exit_reason": "time_exit",
        "holding_days": (exit_dt - entry_dt).days + 1
    }

def main():
    filt = load_scored(SCORED_FILE)
    if filt.empty:
        print(f"[ERROR] No rows pass threshold {PROBA_THRESHOLD}.")
        return

    min_dt, max_dt = filt["published_dt"].min(), filt["published_dt"].max()
    prices = fetch_prices(filt["ticker"].tolist(), min_dt, max_dt)

    rows = []
    for _, r in filt.iterrows():
        t, d, p = r["ticker"], r["published_dt"], r["proba_best"]
        sim = simulate_time_exit(prices.get(t), d)
        if sim is None:
            continue
        sim.update({"ticker": t, "published_dt": d, "proba_best": float(p)})
        rows.append(sim)

    trades = pd.DataFrame(rows).sort_values(["published_dt","ticker"])
    Path(TRADES_OUT).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES_OUT, index=False)

    n = len(trades)
    if n > 0:
        wr = (trades["net_return"] > 0).mean()
        avg = trades["net_return"].mean()
        med = trades["net_return"].median()
        std = trades["net_return"].std(ddof=1)
        sharpe = (avg / std) if (std and std > 0) else np.nan
        tot = trades["net_return"].sum()
    else:
        wr = avg = med = sharpe = tot = np.nan

    summary = pd.DataFrame([{
        "trades": n,
        "win_rate": round(float(wr), 4) if pd.notna(wr) else np.nan,
        "avg_trade_return": round(float(avg), 6) if pd.notna(avg) else np.nan,
        "median_trade_return": round(float(med), 6) if pd.notna(med) else np.nan,
        "sharpe_per_trade": round(float(sharpe), 4) if pd.notna(sharpe) else np.nan,
        "total_return_sum": round(float(tot), 6) if pd.notna(tot) else np.nan,
        "holding_days": HOLDING_DAYS,
        "cost_bps": COST_BPS,
        "proba_threshold": PROBA_THRESHOLD
    }])
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"[OK] ML‑filtered trades: {len(trades)} → {TRADES_OUT}")
    print(f"[OK] Summary → {SUMMARY_OUT}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

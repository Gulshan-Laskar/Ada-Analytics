# backtest_signals_timeexit.py
# Minimal backtester: buy next day's OPEN, hold N trading days, exit at CLOSE.
# No stop-loss, no take-profit.
# Requirements: pip install pandas numpy yfinance

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# ========= EDIT THESE PATHS =========
SIGNALS_FILE = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/signals.csv"            # must contain at least: published_dt, ticker
TRADES_OUT   = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/trades_backtest.csv"    # per-trade results
SUMMARY_OUT  = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/backtest_summary.csv"   # one-line summary metrics
# ====================================

# ========= TUNE THESE ===============
HOLDING_DAYS = 5        # fixed holding period (trading days)
COST_BPS     = 10       # round-trip cost in basis points (e.g. 10 = 0.10%)
# ====================================

def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "published_dt" not in df.columns or "ticker" not in df.columns:
        raise ValueError("signals.csv must contain at least 'published_dt' and 'ticker'.")
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["published_dt", "ticker"])
    df = df.drop_duplicates(subset=["ticker", "published_dt"]).reset_index(drop=True)
    return df

def fetch_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Download daily OHLC for each ticker with a small buffer around the signal window."""
    start_buf = start - pd.Timedelta(days=10)
    end_buf   = end + pd.Timedelta(days=10)
    data = {}
    for t in sorted(set(tickers)):
        try:
            hist = yf.download(
                t,
                start=start_buf.strftime("%Y-%m-%d"),
                end=end_buf.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,  # explicit to avoid warnings; keeps raw OHLC
                actions=False,
                interval="1d",
            )
            if not hist.empty:
                hist = hist[["Open","High","Low","Close"]].copy()
                hist.index = pd.to_datetime(hist.index)
                # If any duplicate dates occur, keep the first row
                hist = hist.groupby(level=0).first().sort_index()
                data[t] = hist
        except Exception as e:
            print(f"[WARN] Failed to fetch {t}: {e}")
    return data

def next_trading_day(prices: pd.DataFrame, after_date: pd.Timestamp) -> pd.Timestamp | None:
    """Find the first trading date strictly after 'after_date'."""
    idx = prices.index
    pos = idx.searchsorted(after_date + pd.Timedelta(days=1))
    if pos < len(idx):
        return idx[pos]
    return None

def _scalar(v) -> float | None:
    """Coerce a scalar/Series cell to float; return None if not finite/valid."""
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except Exception:
        return None

def simulate_trade_time_exit(prices: pd.DataFrame, signal_date: pd.Timestamp) -> dict:
    """Buy next-day OPEN, exit at CLOSE after HOLDING_DAYS. No TP/SL."""
    if prices is None or prices.empty:
        return {"skipped": True, "reason": "no_prices"}

    entry_date = next_trading_day(prices, signal_date)
    if entry_date is None or entry_date not in prices.index:
        return {"skipped": True, "reason": "no_entry"}

    entry_open = _scalar(prices.loc[entry_date, "Open"])
    if entry_open is None:
        return {"skipped": True, "reason": "bad_entry_price"}

    all_dates = prices.index
    start_pos = all_dates.get_loc(entry_date)
    end_pos   = min(start_pos + HOLDING_DAYS - 1, len(all_dates) - 1)

    exit_date = all_dates[end_pos]
    exit_close = _scalar(prices.loc[exit_date, "Close"])
    if exit_close is None:
        return {"skipped": True, "reason": "bad_exit_price"}

    gross_ret = (exit_close - entry_open) / entry_open
    net_ret = gross_ret - (COST_BPS / 10000.0)

    return {
        "skipped": False,
        "entry_date": entry_date,
        "entry_price": float(entry_open),
        "exit_date": exit_date,
        "exit_price": float(exit_close),
        "gross_return": float(gross_ret),
        "net_return": float(net_ret),
        "exit_reason": "time_exit",
        "holding_days": (exit_date - entry_date).days + 1
    }

def main():
    Path(TRADES_OUT).parent.mkdir(parents=True, exist_ok=True)

    signals = load_signals(SIGNALS_FILE)
    if signals.empty:
        print("[ERROR] No signals found after cleaning.")
        return

    min_dt, max_dt = signals["published_dt"].min(), signals["published_dt"].max()
    tickers = signals["ticker"].tolist()
    print(f"[INFO] Signals: {len(signals)} rows | tickers={signals['ticker'].nunique()} | "
          f"range={min_dt.date()} → {max_dt.date()}")

    prices_map = fetch_prices(tickers, min_dt, max_dt)

    rows = []
    for _, r in signals.iterrows():
        tkr, sig_dt = r["ticker"], r["published_dt"]
        px = prices_map.get(tkr)
        sim = simulate_trade_time_exit(px, sig_dt)
        out = {"ticker": tkr, "published_dt": sig_dt}
        out.update(sim)
        rows.append(out)

    trades = pd.DataFrame(rows)
    executed = trades[~trades["skipped"]].copy().sort_values(["ticker","entry_date"])
    executed.to_csv(TRADES_OUT, index=False)

    # Summary metrics
    n = len(executed)
    if n > 0:
        win_rate = (executed["net_return"] > 0).mean()
        avg_ret  = executed["net_return"].mean()
        med_ret  = executed["net_return"].median()
        std_ret  = executed["net_return"].std(ddof=1)
        sharpe   = (avg_ret / std_ret) if (std_ret and std_ret > 0) else np.nan
        tot_ret  = executed["net_return"].sum()
    else:
        win_rate = avg_ret = med_ret = sharpe = tot_ret = np.nan

    summary = pd.DataFrame([{
        "trades": n,
        "win_rate": round(float(win_rate), 4) if pd.notna(win_rate) else np.nan,
        "avg_trade_return": round(float(avg_ret), 6) if pd.notna(avg_ret) else np.nan,
        "median_trade_return": round(float(med_ret), 6) if pd.notna(med_ret) else np.nan,
        "sharpe_per_trade": round(float(sharpe), 4) if pd.notna(sharpe) else np.nan,
        "total_return_sum": round(float(tot_ret), 6) if pd.notna(tot_ret) else np.nan,
        "holding_days": HOLDING_DAYS,
        "cost_bps": COST_BPS
    }])
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"[OK] Saved trades to: {Path(TRADES_OUT).resolve()}")
    print(f"[OK] Saved summary to: {Path(SUMMARY_OUT).resolve()}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

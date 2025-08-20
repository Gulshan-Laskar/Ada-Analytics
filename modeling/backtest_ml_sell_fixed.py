# modeling/backtest_ml_sell_fixed.py
# Fixed ML-entry + TP/SL exit backtest (no sweep).
# Paths are repo-relative. Price cache stored at data/price_cache.pkl.
#
# Run:
#   cd <repo-root>
#   python -m modeling.backtest_ml_sell_fixed

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import pickle
import sys

# ----------------- Repo paths -----------------
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent                # repo root
DATA = ROOT / "data"

SCORED_FILE = DATA / "scored_test.csv"   # needs: published_dt, ticker, proba_best
TRADES_OUT  = DATA / "trades_backtest_ml.csv"
SUMMARY_OUT = DATA / "backtest_summary_ml.csv"
PRICE_CACHE = DATA / "price_cache.pkl"
# ----------------------------------------------

# ------------- Your locked settings -----------
FILTER_MODE       = "threshold"   # only threshold used here
PROBA_THRESHOLD   = 0.75
EXIT_MODE         = "tp_sl"       # "tp_sl" only in this fixed file
HOLDING_DAYS      = 7
TP_PCT            = 0.08
SL_PCT            = 0.05
TP_FIRST          = True          # check TP before SL each day
COST_BPS          = 10
MAX_POS_PER_DAY   = None          # set int to cap positions per day, or None
# ----------------------------------------------

def _scalar(v):
    """Coerce a cell (scalar/Series) to finite positive float or None."""
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except Exception:
        return None

def load_scored(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] Missing scored file: {path}")
    df = pd.read_csv(path)
    need = {"published_dt", "ticker", "proba_best"}
    if not need.issubset(df.columns):
        sys.exit(f"[ERROR] {path} must contain columns: {sorted(need)}")
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["published_dt", "ticker", "proba_best"])
    # dedupe per (ticker, disclosure)
    df = df.drop_duplicates(subset=["ticker","published_dt"]).sort_values("published_dt").reset_index(drop=True)
    # entry filter
    df = df[df["proba_best"] >= PROBA_THRESHOLD].copy()
    if MAX_POS_PER_DAY is not None:
        df = (df.sort_values(["published_dt","proba_best"], ascending=[True,False])
                .groupby("published_dt", group_keys=False).head(MAX_POS_PER_DAY)
                .reset_index(drop=True))
    return df

def fetch_prices_yf(tickers, start, end):
    """Download daily OHLC per ticker with buffers; dedupe & sanitize."""
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
                px = hist[["Open","High","Low","Close"]].copy()
                px.index = pd.to_datetime(px.index, errors="coerce")
                if getattr(px.index, "tz", None) is not None:
                    px.index = px.index.tz_localize(None)
                px = px[~px.index.isna()]
                px = px.groupby(level=0).first().sort_index()
                data[t] = px
        except Exception as e:
            print(f"[WARN] Failed {t}: {e}")
    return data

def load_or_build_price_cache(tickers, start, end) -> dict:
    cache = {}
    if PRICE_CACHE.exists():
        try:
            with open(PRICE_CACHE, "rb") as fh:
                cache = pickle.load(fh)
        except Exception:
            cache = {}
    need = set(map(str.upper, tickers))
    have = set(cache.keys())
    missing = sorted(list(need - have))
    if missing:
        fresh = fetch_prices_yf(missing, start, end)
        cache.update(fresh)
        try:
            PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(PRICE_CACHE, "wb") as fh:
                pickle.dump(cache, fh)
            print(f"[OK] Updated price cache → {PRICE_CACHE} (+{len(fresh)} tickers)")
        except Exception as e:
            print(f"[WARN] Could not write price cache: {e}")
    else:
        print(f"[OK] Using cached prices ({len(cache)} tickers) -> {PRICE_CACHE}")
    return cache

def next_trading_day(idx: pd.DatetimeIndex, after: pd.Timestamp):
    pos = idx.searchsorted(after + pd.Timedelta(days=1))
    return idx[pos] if pos < len(idx) else None

def exit_tp_sl(px: pd.DataFrame, entry_dt, entry_open, holding_days, tp_pct, sl_pct, tp_first=True):
    """Intraday TP/SL using High/Low; else exit at time."""
    tp_level = entry_open * (1 + tp_pct)
    sl_level = entry_open * (1 - sl_pct)
    all_dates = px.index
    pos0 = all_dates.get_loc(entry_dt)
    pos1 = min(pos0 + holding_days - 1, len(all_dates) - 1)

    for pos in range(pos0, pos1 + 1):
        d = all_dates[pos]
        hi = _scalar(px.loc[d, "High"])
        lo = _scalar(px.loc[d, "Low"])
        if hi is None or lo is None:
            continue
        if tp_first:
            if hi >= tp_level: return d, tp_level, "take_profit"
            if lo <= sl_level: return d, sl_level, "stop_loss"
        else:
            if lo <= sl_level: return d, sl_level, "stop_loss"
            if hi >= tp_level: return d, tp_level, "take_profit"

    # time exit at close if no TP/SL hit
    exit_dt = all_dates[pos1]
    exit_close = _scalar(px.loc[exit_dt, "Close"])
    if exit_close is None: return None
    return exit_dt, exit_close, "time_exit"

def simulate_trade(px: pd.DataFrame, published_dt: pd.Timestamp):
    """Buy next trading day OPEN; exit via TP/SL or time with costs deducted."""
    if px is None or px.empty:
        return None
    entry_dt = next_trading_day(px.index, published_dt)
    if entry_dt is None or entry_dt not in px.index:
        return None
    entry_open = _scalar(px.loc[entry_dt, "Open"])
    if entry_open is None:
        return None

    res = exit_tp_sl(px, entry_dt, entry_open, HOLDING_DAYS, TP_PCT, SL_PCT, TP_FIRST)
    if res is None:
        return None

    exit_dt, exit_price, reason = res
    gross = (exit_price - entry_open) / entry_open
    net = gross - (COST_BPS / 10000.0)
    return {
        "entry_date": entry_dt,
        "entry_price": float(entry_open),
        "exit_date": exit_dt,
        "exit_price": float(exit_price),
        "gross_return": float(gross),
        "net_return": float(net),
        "exit_reason": reason,
        "holding_days": (exit_dt - entry_dt).days + 1
    }

def main():
    selected = load_scored(SCORED_FILE)
    if selected.empty:
        print("[ERROR] No rows pass threshold filter.")
        # still write empty outputs for robustness
        pd.DataFrame().to_csv(TRADES_OUT, index=False)
        pd.DataFrame().to_csv(SUMMARY_OUT, index=False)
        return

    min_dt, max_dt = selected["published_dt"].min(), selected["published_dt"].max()
    px_map = load_or_build_price_cache(selected["ticker"].unique().tolist(), min_dt, max_dt)

    rows = []
    for _, r in selected.iterrows():
        t, d, p = r["ticker"], r["published_dt"], r["proba_best"]
        sim = simulate_trade(px_map.get(t), d)
        if sim is None:
            continue
        sim.update({"ticker": t, "published_dt": d, "proba_best": float(p)})
        rows.append(sim)

    trades = pd.DataFrame(rows).sort_values(["published_dt","ticker"])
    TRADES_OUT.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES_OUT, index=False)

    # ---- summary ----
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
        "sharpe_per_trade": round(float(sharpe), 6) if pd.notna(sharpe) else np.nan,
        "total_return_sum": round(float(tot), 6) if pd.notna(tot) else np.nan,
        "filter_mode": FILTER_MODE,
        "proba_threshold": PROBA_THRESHOLD,
        "exit_mode": EXIT_MODE,
        "holding_days": HOLDING_DAYS,
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "cost_bps": COST_BPS
    }])
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"[OK] Trades -> {TRADES_OUT}  (n={n})")
    print(f"[OK] Summary -> {SUMMARY_OUT}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

# modeling/backtest_ml_sell_fixed.py
# Fixed ML-entry + TP/SL exit backtest (no sweep) with robust price cache repair.
# Run:
#   cd <repo-root>   # e.g., C:\Users\pusap\OneDrive\Desktop\project\Ada-Analytics
#   python -m modeling.backtest_ml_sell_fixed

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import pickle
import sys

# ----------------- Repo paths -----------------
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent                # repo root: ...\Ada-Analytics
DATA = ROOT / "data"

SCORED_FILE = DATA / "scored_test.csv"   # needs: published_dt, ticker, proba_best
TRADES_OUT  = DATA / "trades_backtest_ml.csv"
SUMMARY_OUT = DATA / "backtest_summary_ml.csv"
PRICE_CACHE = DATA / "price_cache.pkl"
# ----------------------------------------------

# ------------- Locked settings ----------------
PROBA_THRESHOLD = 0.75
HOLDING_DAYS    = 7
TP_PCT          = 0.08
SL_PCT          = 0.05
TP_FIRST        = True
COST_BPS        = 10
MAX_POS_PER_DAY = None   # e.g., 10 to cap positions per day; None = no cap
# ----------------------------------------------

def _scalar(v):
    """Coerce a cell (scalar/Series) to finite float or None."""
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) else None
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
    df = df.drop_duplicates(subset=["ticker","published_dt"]).sort_values("published_dt").reset_index(drop=True)
    df = df[df["proba_best"] >= PROBA_THRESHOLD].copy()
    if MAX_POS_PER_DAY is not None:
        df = (df.sort_values(["published_dt","proba_best"], ascending=[True,False])
                .groupby("published_dt", group_keys=False).head(MAX_POS_PER_DAY)
                .reset_index(drop=True))
    return df

def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns Open/High/Low/Close exist. If missing, synthesize conservatively."""
    cols = df.columns
    out = df.copy()
    if "Close" not in cols and "Adj Close" in cols:
        out["Close"] = out["Adj Close"]
    if "Open" not in out.columns and "Close" in out.columns:
        out["Open"] = out["Close"]
    if "High" not in out.columns and "Close" in out.columns:
        out["High"] = out["Close"]
    if "Low"  not in out.columns and "Close" in out.columns:
        out["Low"]  = out["Close"]
    keep = [c for c in ["Open","High","Low","Close"] if c in out.columns]
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.isna()]
    return out.groupby(level=0).first().sort_index()

def yf_fetch_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    try:
        hist = yf.download(
            ticker,
            start=(start - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            end=(end   + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=False,
            actions=False,
        )
        if hist is None or hist.empty:
            return None
        return ensure_ohlc(hist)
    except Exception:
        return None

def load_or_build_price_cache(tickers, start, end) -> dict:
    """Load cache; refetch missing or malformed (no High/Low) tickers; repair if needed."""
    cache: dict[str, pd.DataFrame] = {}
    if PRICE_CACHE.exists():
        try:
            with open(PRICE_CACHE, "rb") as fh:
                cache = pickle.load(fh)
        except Exception:
            cache = {}

    need = set(map(str.upper, tickers))
    to_check = sorted(list(need))
    updated = 0

    for t in to_check:
        df = cache.get(t)
        if df is None or not isinstance(df, pd.DataFrame) or not {"Open","High","Low","Close"}.issubset(df.columns):
            refetched = yf_fetch_one(t, start, end)
            if refetched is not None and not refetched.empty:
                cache[t] = refetched
                updated += 1
            else:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    cache[t] = ensure_ohlc(df)
                else:
                    cache[t] = pd.DataFrame(columns=["Open","High","Low","Close"])

    if updated > 0 or not PRICE_CACHE.exists():
        try:
            PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(PRICE_CACHE, "wb") as fh:
                pickle.dump(cache, fh)
            print(f"[OK] Price cache repaired/updated (+{updated}) -> {PRICE_CACHE}")
        except Exception as e:
            print(f"[WARN] Could not write price cache: {e}")
    else:
        print(f"[OK] Using cached prices ({len(cache)} tickers) -> {PRICE_CACHE}")

    for t in list(cache.keys()):
        cache[t] = ensure_ohlc(cache[t])

    return cache

def next_trading_day(idx: pd.DatetimeIndex, after: pd.Timestamp):
    pos = idx.searchsorted(after + pd.Timedelta(days=1))
    return idx[pos] if pos < len(idx) else None

def exit_tp_sl(px: pd.DataFrame, entry_dt, entry_open, holding_days, tp_pct, sl_pct, tp_first=True):
    """Intraday TP/SL using High/Low; else exit at time."""
    if px is None or px.empty or not {"High","Low","Close"}.issubset(px.columns):
        return None
    tp_level = entry_open * (1 + tp_pct)
    sl_level = entry_open * (1 - sl_pct)
    all_dates = px.index
    if entry_dt not in all_dates:
        return None
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

    exit_dt = all_dates[pos1]
    exit_close = _scalar(px.loc[exit_dt, "Close"])
    if exit_close is None: return None
    return exit_dt, exit_close, "time_exit"

def simulate_trade(px: pd.DataFrame, published_dt: pd.Timestamp):
    """Buy next trading day OPEN; exit via TP/SL or time; subtract costs."""
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
        # Fallback: time exit on HOLDING_DAYS using Close.
        all_dates = px.index
        pos0 = all_dates.get_loc(entry_dt)
        pos1 = min(pos0 + HOLDING_DAYS - 1, len(all_dates) - 1)
        exit_dt = all_dates[pos1]
        exit_close = _scalar(px.loc[exit_dt, "Close"])
        if exit_close is None:
            return None
        gross = (exit_close - entry_open) / entry_open
        net = gross - (COST_BPS / 10000.0)
        return {
            "entry_date": entry_dt,
            "entry_price": float(entry_open),
            "exit_date": exit_dt,
            "exit_price": float(exit_close),
            "gross_return": float(gross),
            "net_return": float(net),
            "exit_reason": "time_exit_fallback",
            "holding_days": (exit_dt - entry_dt).days + 1
        }

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
        "proba_threshold": PROBA_THRESHOLD,
        "exit_mode": "tp_sl",
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

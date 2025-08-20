import re
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from collections import Counter

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
SIGNALS_FILE = BASE_DIR / "data" / "signals.csv"
ML_OUT = BASE_DIR / "data" / "ml_dataset.csv"
SKIPPED_OUT = BASE_DIR / "data" / "ml_skipped_symbols.xlsx"

VALID_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\-\.]{0,9}$")

def normalize_ticker(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.upper().strip().replace(".", "-")
    return re.sub(r"\s+", "", t)

def looks_valid(t: str) -> bool:
    if not t or len(t) > 10:
        return False
    return bool(VALID_TICKER_RE.match(t))

def safe_download(ticker: str, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, actions=False, interval="1d")
        if df is None or df.empty:
            return None
        df = df[["Open","Close"]].copy()
        df.index = pd.to_datetime(df.index)
        return df.groupby(level=0).first().sort_index()
    except Exception:
        return None

def next_trading_day(idx: pd.DatetimeIndex, after: pd.Timestamp):
    pos = idx.searchsorted(after + pd.Timedelta(days=1))
    return idx[pos] if pos < len(idx) else None

def to_scalar(x):
    if isinstance(x, pd.Series):
        x = x.iloc[0] if len(x) else np.nan
    try:
        f = float(x)
        return f if np.isfinite(f) and f > 0 else None
    except Exception:
        return None

def forward_return_series(px: pd.DataFrame, published_dt: pd.Timestamp, horizons=(1,3,5,10)):
    out = {}
    if px is None or px.empty or pd.isna(published_dt):
        return out
    entry_date = next_trading_day(px.index, published_dt)
    if entry_date is None:
        return out
    e_open = to_scalar(px.loc[entry_date, "Open"])
    if e_open is None:
        return out
    pos0 = px.index.get_loc(entry_date)
    for h in horizons:
        pos1 = pos0 + h - 1
        if pos1 < len(px.index):
            d1 = px.index[pos1]
            c1 = to_scalar(px.loc[d1, "Close"])
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
        print("[ERROR] No valid signals after filtering.")
        return

    min_dt, max_dt = sig_ok["published_dt"].min(), sig_ok["published_dt"].max()
    px_map = {}
    skipped_tickers = []
    for t in sorted(sig_ok["ticker"].unique()):
        px = safe_download(t, min_dt - pd.Timedelta(days=15), max_dt + pd.Timedelta(days=15))
        if px is None:
            skipped_tickers.append(t)
            continue
        px_map[t] = px

    rows = []
    skipped_rows = []
    keep_cols = [c for c in ["published_dt","ticker","company","politician_name","type","owner","sector","size_avg_usd","trade_size_vs_market_cap","sentiment_score","mention_count","volume_spike_ratio","consensus_score_7d","signal_strength"] if c in sig_ok.columns]

    for _, r in sig_ok.iterrows():
        t = r["ticker"]
        px = px_map.get(t)
        fwd = forward_return_series(px, r["published_dt"])
        if not fwd:
            skipped_rows.append({"ticker": t, "published_dt": r["published_dt"], "reason": "no_forward_prices"})
            continue
        row = r[keep_cols].to_dict()
        row.update(fwd)
        rows.append(row)

    out = pd.DataFrame(rows)
    ML_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ML_OUT, index=False)

    with pd.ExcelWriter(SKIPPED_OUT) as xw:
        pd.DataFrame({"ticker": skipped_tickers}).to_excel(xw, sheet_name="yahoo_empty", index=False)
        pd.DataFrame(skipped_rows).to_excel(xw, sheet_name="no_forward_prices", index=False)

    print(f"[OK] Wrote ML dataset: {ML_OUT} with shape {out.shape}")
    print(f"[OK] Wrote skipped report: {SKIPPED_OUT}")

if __name__ == "__main__":
    main()

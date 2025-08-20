# label_for_ml_robust.py
# Create ML labels (forward returns) from signals, robust to bad tickers & duplicate price rows.
# pip install pandas numpy yfinance

import re
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from collections import Counter

# ==== EDIT ====
SIGNALS_FILE = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/signals.csv"
ML_OUT       = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/ml_dataset.csv"
SKIPPED_OUT  = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/ml_skipped_symbols.csv"
# ==============

# Accept US-like tickers: letters/digits plus optional single '-' (Yahoo uses BRK-B), length <= 6 by default
VALID_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\-\.]{0,9}$")  # allow '.' too; we'll map '.'->'-' for Yahoo

def normalize_ticker(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.upper().strip()
    # Common Yahoo quirk: class shares like BRK.B are BRK-B on Yahoo
    if "." in t and t.count(".") == 1 and len(t) <= 8:
        t = t.replace(".", "-")
    # Remove spaces and weird unicode
    t = re.sub(r"\s+", "", t)
    return t

def looks_valid(t: str) -> bool:
    # eliminate long words or phrases that slipped into ticker column
    if not t or len(t) > 10:
        return False
    if not VALID_TICKER_RE.match(t):
        return False
    # disallow strings that are clearly non-tickers (heuristics)
    if "-" not in t and "." not in t and len(t) >= 7:
        # long, unseparated strings (e.g., LIMITEDIREN)
        return False
    return True

def safe_download(ticker: str, start, end):
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            actions=False,
            interval="1d",
        )
        if df is None or df.empty:
            return None
        df = df[["Open","Close"]].copy()
        df.index = pd.to_datetime(df.index)
        # Deduplicate any repeated dates
        df = df.groupby(level=0).first().sort_index()
        return df
    except Exception as e:
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
    # align datatypes
    if "published_dt" not in sig.columns or "ticker" not in sig.columns:
        raise ValueError("signals.csv must include 'published_dt' and 'ticker'")
    sig["published_dt"] = pd.to_datetime(sig["published_dt"], errors="coerce")
    # prefer 'ticker' else fallback to 'ticker_original'
    if sig["ticker"].isna().all() and "ticker_original" in sig.columns:
        sig["ticker"] = sig["ticker_original"]
    sig["ticker"] = sig["ticker"].astype(str).map(normalize_ticker)

    # try to recover obviously broken tickers from 'ticker_original'
    if "ticker_original" in sig.columns:
        m = sig["ticker"].eq("") & sig["ticker_original"].notna()
        sig.loc[m, "ticker"] = sig.loc[m, "ticker_original"].astype(str).map(normalize_ticker)

    # filter obviously invalid symbols
    valid_mask = sig["ticker"].map(looks_valid)
    skipped_reason = []
    skipped_rows = []

    # keep a copy of original for features
    keep_cols = [
        "published_dt","ticker","company","politician_name","type","owner","sector",
        "size_avg_usd","trade_size_vs_market_cap","sentiment_score","mention_count",
        "volume_spike_ratio","consensus_score_7d","signal_strength"
    ]
    keep_cols = [c for c in keep_cols if c in sig.columns]

    sig_ok = sig[valid_mask & sig["published_dt"].notna()].copy()
    # de-dup on ticker + published_dt
    sig_ok = sig_ok.drop_duplicates(subset=["ticker","published_dt"]).reset_index(drop=True)

    if sig_ok.empty:
        print("[ERROR] No valid signals after ticker normalization/filtering.")
        Path(SKIPPED_OUT).write_text("no valid signals")
        return

    min_dt, max_dt = sig_ok["published_dt"].min(), sig_ok["published_dt"].max()

    # download prices per unique ticker
    px_map = {}
    for t in sorted(sig_ok["ticker"].unique()):
        if not looks_valid(t):
            skipped_reason.append(("invalid_regex", t))
            continue
        px = safe_download(t, min_dt - pd.Timedelta(days=15), max_dt + pd.Timedelta(days=15))
        if px is None or px.empty:
            skipped_reason.append(("yahoo_empty", t))
            continue
        px_map[t] = px

    # build ML rows
    rows = []
    for _, r in sig_ok.iterrows():
        t = r["ticker"]
        px = px_map.get(t)
        fwd = forward_return_series(px, r["published_dt"])
        if not fwd:
            skipped_rows.append({"ticker": t, "published_dt": r["published_dt"], "reason": "no_forward_prices"})
            continue
        row = {k: r[k] for k in keep_cols}
        row.update(fwd)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(ML_OUT, index=False)

    # write skipped summary
    skipped_df = pd.DataFrame(skipped_rows)
    sym_counts = Counter([sym for _, sym in skipped_reason])
    reasons_df = pd.DataFrame(skipped_reason, columns=["reason","ticker"]).drop_duplicates()
    with pd.ExcelWriter(SKIPPED_OUT.replace(".csv",".xlsx")) as xw:
        pd.DataFrame.from_dict(sym_counts, orient="index", columns=["count"]).sort_values("count", ascending=False).to_excel(xw, sheet_name="symbols_skipped")
        reasons_df.to_excel(xw, sheet_name="reasons", index=False)
        if not skipped_df.empty:
            skipped_df.to_excel(xw, sheet_name="rows_no_forward_prices", index=False)

    print(f"[OK] wrote ML dataset: {Path(ML_OUT).resolve()} with shape {out.shape}")
    print(f"[OK] wrote skipped report: {Path(SKIPPED_OUT).replace('.csv','.xlsx')}")
    kept = len(sig_ok)
    print(f"[INFO] signals considered: {kept} | tickers with prices: {len(px_map)} | rows in ML set: {len(out)}")

if __name__ == "__main__":
    main()

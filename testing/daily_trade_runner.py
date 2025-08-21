# execution/daily_trade_runner.py
# Daily BUY/SELL order generation + optional Alpaca submission (paper).
# - Buys: latest disclosure date ≤ TODAY, proba_best ≥ 0.75, equal-weight sizing
# - Sells: TP=+8%, SL=-5%, max hold=7 trading days, from open_positions.csv
# - Outputs Alpaca-ready CSVs (buy + sell) and submits if SUBMIT_TO_ALPACA=True
#
# Run:
#   cd <repo-root>
#   python -m execution.daily_trade_runner

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import yfinance as yf

# Optional Alpaca submission
try:
    from alpaca_trade_api.rest import REST
    HAVE_ALPACA = True
except Exception:
    HAVE_ALPACA = False

# ----------------- Repo paths -----------------
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent                # repo root
DATA = ROOT / "data"
OUT = ROOT / "data"                      # write orders here

SCORED_FILE   = DATA / "scored_test.csv"         # published_dt, ticker, proba_best
OPEN_POS_FILE = DATA / "open_positions.csv"      # ticker, entry_date, entry_price, shares

# ----------------- Settings -------------------
TODAY = datetime.now().strftime("%Y-%m-%d")      # or set fixed date string if testing
THRESHOLD = 0.75
MAX_CONCURRENT = None     # e.g., 10 (caps buys per day) or None
CAPITAL = 100_000         # paper capital allocation
MAX_PER_NAME_PCT = 0.10   # 10% cap per ticker
WEIGHTING = "equal"       # "equal" or "proba"

# Sell rule (fixed)
TP_PCT = 0.08
SL_PCT = 0.05
MAX_HOLD_DAYS = 7

# Alpaca submission
SUBMIT_TO_ALPACA = False  # set True to actually submit
PAPER = True              # paper account
# ----------------------------------------------

def _scalar(v):
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"published_dt","ticker","proba_best"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain: {sorted(need)}")
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["published_dt","ticker","proba_best"])
    # pick the latest available disclosure date ≤ TODAY
    today = pd.to_datetime(TODAY).normalize()
    df = df[df["published_dt"] <= today]
    if df.empty:
        return df
    latest = df["published_dt"].max()
    df = df[df["published_dt"] == latest].copy()
    # threshold + optional top-K
    df = df[df["proba_best"] >= THRESHOLD].copy()
    df = df.sort_values("proba_best", ascending=False)
    if MAX_CONCURRENT:
        df = df.head(MAX_CONCURRENT)
    return df.reset_index(drop=True)

def fetch_last_price_yf(symbol: str) -> float | None:
    try:
        bar = yf.download(symbol, period="5d", interval="1d", progress=False).tail(1)
        if bar is None or bar.empty:
            return None
        return _scalar(bar["Close"].iloc[-1])
    except Exception:
        return None

def build_buy_orders(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["symbol","qty","side","type","time_in_force",
                                     "take_profit_price","stop_loss_price"])
    # weights
    if WEIGHTING == "proba":
        w = np.maximum(candidates["proba_best"].values - THRESHOLD, 0.0)
        if w.sum() == 0:
            w = np.ones(len(candidates))
    else:
        w = np.ones(len(candidates))
    weights = w / w.sum()

    # fetch prices and allocate
    prices = []
    for t in candidates["ticker"]:
        p = fetch_last_price_yf(t)
        prices.append(p)
    cand = candidates.copy()
    cand["last_price"] = prices
    cand = cand.dropna(subset=["last_price"])
    if cand.empty:
        return pd.DataFrame(columns=["symbol","qty","side","type","time_in_force",
                                     "take_profit_price","stop_loss_price"])

    cap_per = CAPITAL * MAX_PER_NAME_PCT
    allocs = np.minimum(weights[:len(cand)] * CAPITAL, cap_per)
    cand["alloc_usd"] = np.round(allocs, 2)
    cand["qty"] = np.floor(cand["alloc_usd"] / cand["last_price"]).astype(int)
    cand = cand[cand["qty"] > 0].copy()
    if cand.empty:
        return pd.DataFrame(columns=["symbol","qty","side","type","time_in_force",
                                     "take_profit_price","stop_loss_price"])

    # compute bracket price levels from last_price (approx)
    cand["take_profit_price"] = np.round(cand["last_price"] * (1 + TP_PCT), 4)
    cand["stop_loss_price"]   = np.round(cand["last_price"] * (1 - SL_PCT), 4)

    orders = cand.assign(
        symbol=cand["ticker"],
        side="buy",
        type="market",
        time_in_force="day"
    )[["symbol","qty","side","type","time_in_force","take_profit_price","stop_loss_price"]]
    return orders

def load_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker","entry_date","entry_price","shares"])
    df = pd.read_csv(path)
    need = {"ticker","entry_date","entry_price","shares"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain: {sorted(need)}")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").astype("Int64")
    return df.dropna(subset=["ticker","entry_date","entry_price","shares"]).reset_index(drop=True)

def fetch_ohlc(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            start=(start - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
        )
        if df is None or df.empty:
            return None
        out = df[["Open","High","Low","Close"]].copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        if getattr(out.index, "tz", None) is not None:
            out.index = out.index.tz_localize(None)
        return out.groupby(level=0).first().sort_index()
    except Exception:
        return None

def next_trading_day(idx: pd.DatetimeIndex, after: pd.Timestamp):
    pos = idx.searchsorted(after + pd.Timedelta(days=1))
    return idx[pos] if pos < len(idx) else None

def trading_days_since(idx: pd.DatetimeIndex, entry_dt: pd.Timestamp, up_to: pd.Timestamp) -> int:
    start_dt = entry_dt if entry_dt in idx else next_trading_day(idx, entry_dt - pd.Timedelta(days=1))
    if start_dt is None:
        return 0
    s = idx.searchsorted(start_dt)
    e = idx.searchsorted(up_to, side="right") - 1
    return max(0, e - s + 1)

def build_sell_orders(positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (audit, alpaca_orders)."""
    if positions.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["symbol","qty","side","type","time_in_force"])

    today = pd.to_datetime(TODAY)
    orders = []

    for _, r in positions.iterrows():
        sym = r["ticker"]
        entry_dt = pd.to_datetime(r["entry_date"])
        entry_px = float(r["entry_price"])
        shares = int(r["shares"])

        tp = entry_px * (1 + TP_PCT)
        sl = entry_px * (1 - SL_PCT)

        px = fetch_ohlc(sym, entry_dt, today)
        if px is None or px.empty:
            continue

        reason = None
        hit_date = None
        # check days up to today
        for d in px.index:
            if d < entry_dt or d > today:
                continue
            hi = _scalar(px.loc[d,"High"]); lo = _scalar(px.loc[d,"Low"])
            if hi is None or lo is None:
                continue
            if hi >= tp: reason, hit_date = "take_profit", d; break
            if lo <= sl: reason, hit_date = "stop_loss",   d; break

        if reason is None:
            held_td = trading_days_since(px.index, entry_dt, min(today, px.index.max()))
            if held_td >= MAX_HOLD_DAYS:
                reason, hit_date = "time_exit_day7", today

        if reason is not None:
            orders.append({
                "ticker": sym, "shares": shares, "reason": reason,
                "entry_date": entry_dt.date(), "entry_price": round(entry_px,6),
                "tp_level": round(tp,6), "sl_level": round(sl,6),
                "hit_date": pd.to_datetime(hit_date).date(),
                "suggested_action": "SELL_MKT_TODAY"
            })

    audit = pd.DataFrame(orders)
    if audit.empty:
        return audit, pd.DataFrame(columns=["symbol","qty","side","type","time_in_force"])

    alp = audit.rename(columns={"ticker":"symbol","shares":"qty"})[["symbol","qty"]].copy()
    alp["side"] = "sell"; alp["type"] = "market"; alp["time_in_force"] = "day"
    return audit, alp

# ---------- Alpaca submission ----------
def get_alpaca():
    if not HAVE_ALPACA:
        raise RuntimeError("alpaca-trade-api not installed. pip install alpaca-trade-api")
    key = os.getenv("ALPACA_API_KEY"); sec = os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec: raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY env vars.")
    base = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
    return REST(key_id=key, secret_key=sec, base_url=base)

def submit_buy_brackets(api, orders_df: pd.DataFrame):
    for _, r in orders_df.iterrows():
        sym = r["symbol"]; qty = int(r["qty"])
        tp = float(r["take_profit_price"]); sl = float(r["stop_loss_price"])
        try:
            api.submit_order(
                symbol=sym, qty=qty, side="buy", type="market", time_in_force="day",
                order_class="bracket",
                take_profit={"limit_price": tp},
                stop_loss={"stop_price": sl}
            )
            print(f"[OK] BUY {qty} {sym} with bracket TP={tp}, SL={sl}")
        except Exception as e:
            print(f"[WARN] Failed BUY {sym}: {e}")

def submit_sell_market(api, orders_df: pd.DataFrame):
    for _, r in orders_df.iterrows():
        sym = r["symbol"]; qty = int(r["qty"])
        try:
            api.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
            print(f"[OK] SELL {qty} {sym}")
        except Exception as e:
            print(f"[WARN] Failed SELL {sym}: {e}")

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- BUYS ----------
    scored = load_scored(SCORED_FILE)
    buy_orders = build_buy_orders(scored)
    buy_csv = OUT / f"alpaca_buy_orders_{TODAY}.csv"
    buy_orders.to_csv(buy_csv, index=False)
    print(f"[OK] Buy orders -> {buy_csv} (n={len(buy_orders)})")

    # ---------- SELLS ----------
    positions = load_positions(OPEN_POS_FILE)
    audit, sell_orders = build_sell_orders(positions)
    audit_csv = OUT / f"sell_orders_audit_{TODAY}.csv"
    sell_csv  = OUT / f"alpaca_sell_orders_{TODAY}.csv"
    audit.to_csv(audit_csv, index=False)
    sell_orders.to_csv(sell_csv, index=False)
    print(f"[OK] Sell audit -> {audit_csv} (n={len(audit)})")
    print(f"[OK] Sell orders -> {sell_csv} (n={len(sell_orders)})")

    # ---------- SUBMIT (optional) ----------
    if SUBMIT_TO_ALPACA and (len(buy_orders) or len(sell_orders)):
        api = get_alpaca()
        if len(buy_orders):
            submit_buy_brackets(api, buy_orders)
        if len(sell_orders):
            submit_sell_market(api, sell_orders)
    else:
        print("[INFO] SUBMIT_TO_ALPACA=False (DRY RUN). Review CSVs then submit if desired.")

if __name__ == "__main__":
    main()

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import yfinance as yf
from decimal import Decimal, ROUND_HALF_UP
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Silence yfinance auto_adjust FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Optional Alpaca submission
try:
    from alpaca_trade_api.rest import REST
    HAVE_ALPACA = True
except ImportError:
    HAVE_ALPACA = False

# ----------------- Repo paths -----------------
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
DATA = ROOT / "data"
OUT = ROOT / "data" / "daily_orders"

SCORED_FILE = DATA / "scored_test.csv"
OPEN_POS_FILE = DATA / "open_positions.csv"

# ----------------- Settings -------------------
TODAY = datetime.now().strftime("%Y-%m-%d")
THRESHOLD = 0.75
MAX_CONCURRENT_BUYS = 10
PORTFOLIO_RISK_PER_TRADE = 0.01  # Risk 1% of total portfolio value on each new trade

# Sell rule (fixed)
TP_PCT = 0.08
SL_PCT_FALLBACK = 0.05 # Fallback stop-loss if ATR fails
MAX_HOLD_DAYS = 7

# Alpaca submission
SUBMIT_TO_ALPACA = True
PAPER = True
CANCEL_OPEN_ORDERS_ON_START = True # Set to False to disable canceling orders
# ----------------------------------------------

def _scalar(v):
    if isinstance(v, pd.Series): v = v.iloc[0] if len(v) else np.nan
    try:
        f = float(v); return f if np.isfinite(f) else None
    except (ValueError, TypeError): return None

def tick_for_price(last_price: float) -> float:
    return 0.01 if last_price >= 1.0 else 0.0001

def round_to_tick(price: float, tick: float) -> float:
    q = Decimal(str(tick))
    return float(Decimal(str(price)).quantize(q, rounding=ROUND_HALF_UP))

def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["published_dt"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df.dropna(subset=["published_dt", "ticker", "proba_best"], inplace=True)
    
    today = pd.to_datetime(TODAY).normalize()
    df = df[df["published_dt"] <= today]
    if df.empty: return pd.DataFrame()
    
    latest_date = df["published_dt"].max()
    df = df[df["published_dt"] == latest_date].copy()
    
    df = df[df["proba_best"] >= THRESHOLD].sort_values("proba_best", ascending=False)
    if MAX_CONCURRENT_BUYS:
        df = df.head(MAX_CONCURRENT_BUYS)
    return df.reset_index(drop=True)

def fetch_atr_and_price(symbol: str, period: int = 14):
    """Fetches the latest close price and the Average True Range (ATR) for volatility."""
    try:
        hist = yf.download(symbol, period=f"{period+5}d", interval="1d", progress=False, auto_adjust=True)
        if hist.empty: return None, None
        
        high_low = hist['High'] - hist['Low']
        high_close = np.abs(hist['High'] - hist['Close'].shift())
        low_close = np.abs(hist['Low'] - hist['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        last_price = hist['Close'].iloc[-1]
        
        return _scalar(last_price), _scalar(atr)
    except Exception:
        return None, None

def build_buy_orders(candidates: pd.DataFrame, open_positions: list, buying_power: float, api) -> pd.DataFrame:
    if candidates.empty: return pd.DataFrame()

    candidates = candidates[~candidates['ticker'].isin(open_positions)].copy()
    if candidates.empty:
        print("[INFO] All buy candidates are already in open positions.")
        return pd.DataFrame()

    orders = []
    total_cost = 0
    for _, row in candidates.iterrows():
        ticker = row['ticker']
        
        if api:
            try:
                asset = api.get_asset(ticker)
                if not asset.tradable:
                    print(f"[WARN] {ticker} is not tradable. Skipping.")
                    continue
            except Exception:
                print(f"[WARN] Could not verify asset {ticker}. Skipping.")
                continue

        last_price, atr = fetch_atr_and_price(ticker)
        if last_price is None or atr is None:
            print(f"[WARN] Could not get price/ATR for {ticker}. Skipping.")
            continue

        stop_loss_price = last_price - (2 * atr)
        if stop_loss_price >= last_price:
            stop_loss_price = last_price * (1 - SL_PCT_FALLBACK)

        risk_per_share = last_price - stop_loss_price
        if risk_per_share <= 0: continue

        position_size_usd = buying_power * PORTFOLIO_RISK_PER_TRADE
        qty = int(position_size_usd / risk_per_share)
        
        estimated_cost = qty * last_price
        if total_cost + estimated_cost > buying_power:
            print(f"[INFO] Not enough buying power for {ticker}. Stopping buy order creation.")
            break
        
        if qty <= 0: continue
        total_cost += estimated_cost

        tick = tick_for_price(last_price)
        take_profit_price = round_to_tick(last_price * (1 + TP_PCT), tick)
        stop_loss_price = round_to_tick(stop_loss_price, tick)
        
        orders.append({
            "symbol": ticker, "qty": qty, "side": "buy", "type": "market", "time_in_force": "day",
            "take_profit_price": take_profit_price, "stop_loss_price": stop_loss_price,
            "entry_price_est": last_price
        })

    return pd.DataFrame(orders)

def load_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker","entry_date","entry_price","shares"])
    return pd.read_csv(path, parse_dates=["entry_date"])

def build_sell_orders(positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if positions.empty: 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=positions.columns)
    
    today = pd.to_datetime(TODAY)
    orders, still_open = [], []

    for _, r in positions.iterrows():
        sym, entry_dt, entry_px, shares = r["ticker"], r["entry_date"], r["entry_price"], r["shares"]
        px = yf.download(sym, start=entry_dt, end=today, progress=False, auto_adjust=True)
        if px.empty:
            still_open.append(r.to_dict())
            continue

        tp, sl = entry_px * (1 + TP_PCT), entry_px * (1 - SL_PCT)
        reason, hit_date = None, None

        for d, row in px.iterrows():
            if row['High'] >= tp: reason, hit_date = "take_profit", d; break
            if row['Low'] <= sl: reason, hit_date = "stop_loss", d; break
        
        if reason is None and (today - entry_dt).days >= MAX_HOLD_DAYS:
            reason, hit_date = "time_exit", today

        if reason:
            orders.append({
                "ticker": sym, "shares": shares, "reason": reason, "entry_date": entry_dt.date(),
                "entry_price": entry_px, "hit_date": hit_date.date()
            })
        else:
            still_open.append(r.to_dict())
            
    audit = pd.DataFrame(orders)
    alpaca_sells = pd.DataFrame()
    if not audit.empty:
        alpaca_sells = audit[['ticker', 'shares']].rename(columns={"ticker": "symbol", "shares": "qty"})
        alpaca_sells = alpaca_sells.assign(side="sell", type="market", time_in_force="day")
    
    return audit, alpaca_sells, pd.DataFrame(still_open)

def get_alpaca():
    if not HAVE_ALPACA: raise RuntimeError("alpaca-trade-api not installed.")
    key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec: raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY env vars.")
    return REST(key_id=key, secret_key=sec, base_url="https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets")

def submit_orders(api, orders_df: pd.DataFrame):
    for _, r in orders_df.iterrows():
        try:
            order_data = r.to_dict()
            if r['side'] == 'buy':
                order_data['order_class'] = 'bracket'
                order_data['take_profit'] = {'limit_price': order_data.pop('take_profit_price')}
                order_data['stop_loss'] = {'stop_price': order_data.pop('stop_loss_price')}
                order_data.pop('entry_price_est', None)
            
            api.submit_order(**order_data)
            print(f"[OK] Submitted {r['side'].upper()} {r['qty']} {r['symbol']}")
        except Exception as e:
            print(f"[WARN] Failed {r['side'].upper()} {r['symbol']}: {e}")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api = get_alpaca() if SUBMIT_TO_ALPACA else None

    if api:
        account = api.get_account()
        if account.trading_blocked:
            print("[ERROR] Account is currently blocked from trading.")
            return
        buying_power = float(account.buying_power)
        open_positions = [p.symbol for p in api.list_positions()]
        print(f"[INFO] Alpaca Buying Power: ${buying_power:,.2f} | Open Positions: {len(open_positions)}")
        
        if CANCEL_OPEN_ORDERS_ON_START:
            api.cancel_all_orders()
            print("[INFO] Canceled all existing open orders.")
    else:
        buying_power = 100_000
        open_positions = load_positions(OPEN_POS_FILE)['ticker'].tolist()
        print(f"[INFO] Using default capital. Open Positions from file: {len(open_positions)}")

    existing_pos_df = load_positions(OPEN_POS_FILE)
    sell_audit, sell_orders, still_open_df = build_sell_orders(existing_pos_df)
    
    sell_audit.to_csv(OUT / f"sell_orders_audit_{TODAY}.csv", index=False)
    sell_orders.to_csv(OUT / f"alpaca_sell_orders_{TODAY}.csv", index=False)
    print(f"[OK] Sell audit -> {OUT / f'sell_orders_audit_{TODAY}.csv'} (n={len(sell_audit)})")

    scored = load_scored(SCORED_FILE)
    buy_orders = build_buy_orders(scored, open_positions, buying_power, api)
    buy_orders.to_csv(OUT / f"alpaca_buy_orders_{TODAY}.csv", index=False)
    print(f"[OK] Buy orders -> {OUT / f'alpaca_buy_orders_{TODAY}.csv'} (n={len(buy_orders)})")

    if SUBMIT_TO_ALPACA and api:
        if not sell_orders.empty:
            print("\n--- Submitting SELL orders ---")
            submit_orders(api, sell_orders)
        if not buy_orders.empty:
            print("\n--- Submitting BUY orders ---")
            submit_orders(api, buy_orders)
    else:
        print("\n[INFO] SUBMIT_TO_ALPACA=False (DRY RUN). Review CSVs.")

    dfs_to_concat = []
    if not still_open_df.empty:
        dfs_to_concat.append(still_open_df)
    
    if not buy_orders.empty:
        new_buys_for_pos_file = buy_orders[['symbol', 'entry_price_est', 'qty']].rename(columns={
            'symbol': 'ticker', 'entry_price_est': 'entry_price', 'qty': 'shares'
        })
        new_buys_for_pos_file['entry_date'] = pd.to_datetime(TODAY)
        dfs_to_concat.append(new_buys_for_pos_file)
    
    if dfs_to_concat:
        final_positions = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        final_positions = pd.DataFrame(columns=['ticker', 'entry_date', 'entry_price', 'shares'])
    
    final_positions.to_csv(OPEN_POS_FILE, index=False)
    print(f"\n[OK] Updated open positions file -> {OPEN_POS_FILE} (n={len(final_positions)})")

if __name__ == "__main__":
    main()
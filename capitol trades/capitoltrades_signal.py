import pandas as pd
import yfinance as yf
from datetime import timedelta
from tqdm import tqdm
import os
import logging


logging.basicConfig(
    filename=r'capitol trades\fetch_errors.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    filemode='w'
)

# Assuming your cleaned dataset is in df
def add_trade_direction(df):
    # Define mapping for trade types
    direction_map = {
        'buy': 1,
        'sell': -1,
        'exchange': 0
    }
    
    # Apply the mapping
    df['Trade_Direction'] = df['Type'].map(direction_map)
    
    # Optional: drop rows where Trade_Direction is NaN (unknown type)
    df = df.dropna(subset=['Trade_Direction'])
    
    return df

# Load your cleaned data
df = pd.read_csv(r'capitol trades\cleaned_capitol_trades.csv', parse_dates=['Traded', 'Published'])

df = add_trade_direction(df)

# Helper function to fetch prices
def fetch_prices(ticker, start_date, end_date):
    # Skip if ticker is empty or NaN
    if not isinstance(ticker, str) or pd.isna(ticker):
        logging.info(f"Skipping invalid ticker: {ticker}")
        return pd.Series()

    # Skip if start_date is in the future
    if start_date > pd.Timestamp.today():
        logging.info(f"Skipping future date range for {ticker}")
        return pd.Series()

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        return data['Adj Close']
    except Exception as e:
        logging.info(f"Failed to download {ticker} due to error: {e}")
        return pd.Series()

# Store results
returns_data = []

partial_file_path = r"capitol trades\partial_trades_with_returns.csv"

# If partial file exists, read it and skip processed rows
processed_indices = set()
if os.path.exists(partial_file_path):
    partial_df = pd.read_csv(partial_file_path)
    processed_indices = set(partial_df['Unnamed: 0'])  # Or use a specific column as index

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing trades"):
    if idx in processed_indices:
        continue
    ticker = row['Ticker']
    trade_date = row['Traded']
    disclosed_date = row['Published']
    
    # Set price range window
    start_date = min(trade_date, disclosed_date)
    end_date = max(trade_date + timedelta(days=7), disclosed_date + timedelta(days=7))
    
    prices = fetch_prices(ticker, start_date, end_date)
    
    if prices.empty:
        return_5d_trade = None
        return_5d_disclosed = None
        win_flag = None
    else:
        # Trade date return
        try:
            trade_price = prices.loc[prices.index >= trade_date].iloc[0].item()
            trade_5d_price = prices.loc[prices.index >= trade_date + timedelta(days=5)].iloc[0].item()
            return_5d_trade = (trade_5d_price / trade_price) - 1
        except:
            return_5d_trade = None
        
        # Disclosed date return
        try:
            disclosed_price = prices.loc[prices.index >= disclosed_date].iloc[0].item()
            disclosed_5d_price = prices.loc[prices.index >= disclosed_date + timedelta(days=5)].iloc[0].item()
            return_5d_disclosed = (disclosed_5d_price / disclosed_price) - 1
        except:
            return_5d_disclosed = None
        
        # Win flag
        if row['Trade_Direction'] == 1:
            win_flag = 1 if return_5d_trade is not None and return_5d_trade > 0 else 0
        elif row['Trade_Direction'] == -1:
            win_flag = 1 if return_5d_trade is not None and return_5d_trade < 0 else 0
        else:
            win_flag = None
    
    returns_data.append({
        'return_5d_trade': return_5d_trade,
        'return_5d_disclosed': return_5d_disclosed,
        'win_flag': win_flag
    })
    # Append to partial CSV
    pd.DataFrame([{
        'Unnamed: 0': idx,
        'return_5d_trade': return_5d_trade,
        'return_5d_disclosed': return_5d_disclosed,
        'win_flag': win_flag
    }]).to_csv(partial_file_path, mode='a', header=not os.path.exists(partial_file_path), index=False)

# Merge results back
returns_df = pd.DataFrame(returns_data)
df = pd.concat([df, returns_df], axis=1)

# Save output
df.to_csv(r"capitol trades\trades_with_returns.csv", index=False)


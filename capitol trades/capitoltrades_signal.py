import pandas as pd
import yfinance as yf
from datetime import timedelta


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
df = pd.read_csv(r'C:\Users\india\Desktop\Ada Analytics\Code\Congressional Trade Scraper\cleaned_capitol_trades.csv', parse_dates=['Traded', 'Published'])

df = add_trade_direction(df)

# Helper function to fetch prices
def fetch_prices(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data['Adj Close']
    except:
        return pd.Series()

# Store results
returns_data = []

for idx, row in df.iterrows():
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
            trade_price = prices.loc[prices.index >= trade_date].iloc[0]
            trade_5d_price = prices.loc[prices.index >= trade_date + timedelta(days=5)].iloc[0]
            return_5d_trade = (trade_5d_price / trade_price) - 1
        except:
            return_5d_trade = None
        
        # Disclosed date return
        try:
            disclosed_price = prices.loc[prices.index >= disclosed_date].iloc[0]
            disclosed_5d_price = prices.loc[prices.index >= disclosed_date + timedelta(days=5)].iloc[0]
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

# Merge results back
returns_df = pd.DataFrame(returns_data)
df = pd.concat([df, returns_df], axis=1)

# Save output
df.to_csv("C:\\Users\\india\\Desktop\\Ada Analytics\\Code\\Congressional Trade Scraper\\trades_with_returns.csv", index=False)


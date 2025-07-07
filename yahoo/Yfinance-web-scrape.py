# # #yfinance-web-scrape
# # import yfinance as yf

# # ticker = yf.Ticker("AAPL")

# # # Get historical data
# # hist = ticker.history(period="1mo")
# # print(hist)

# # # Get current price
# # print("Current Price:", ticker.info['regularMarketPrice'])

# # # Get major holders
# # print(ticker.major_holders)

# import yfinance as yf
# import pandas as pd

# # Step 1: Download historical data for AAPL (Apple Inc.)
# ticker = yf.Ticker("AAPL")
# df = ticker.history(period="1y")  # last 1 year latest  data
# df = ticker.history(period="1d")  # last 1 month latest data

# # Step 2: Drop any rows with missing values (NaNs)
# df = df.dropna()

# # Step 3: Reset index so 'Date' becomes a column
# df = df.reset_index()

# # Step 4: Normalize the prices for comparison
# # We normalize Open, High, Low, Close using Min-Max normalization
# price_cols = ['Open', 'High', 'Low', 'Close']
# df[price_cols] = df[price_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# # Optional: Save to CSV
# df.to_csv("AAPL_cleaned_normalized.csv", index=False)

# # Preview the result
# print(df.head())

# import requests
# from bs4 import BeautifulSoup

# url = "https://finance.yahoo.com/most-active"
# headers = {
#     "User-Agent": "Mozilla/5.0",
#     "Accept-Language": "en-US,en;q=0.9"
# }

# response = requests.get(url, headers=headers)
# soup = BeautifulSoup(response.text, 'html.parser')

# # Lists to hold extracted data
# tickers = []
# companies = []

# # Find table rows
# rows = soup.select("table tbody tr")

# for row in rows:
#     cols = row.find_all('td')
#     if len(cols) >= 2:
#         ticker = cols[0].text.strip()
#         company = cols[1].text.strip()
#         tickers.append(ticker)
#         companies.append(company)

# # Check if we got any data
# print("Tickers found:", len(tickers))
# print("Companies found:", len(companies))

# # Safe printing loop
# for i in range(min(len(tickers), len(companies))):
#     print(f"{companies[i]} - {tickers[i]}")

# df = pd.DataFrame({"Company": companies, "Ticker": tickers})
# df.to_csv("yahoo_finance_most_active.csv", index=False)

import yfinance as yf
import pandas as pd
import time
import os

# derive path to the trades CSV and read unique tickers once
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'capitol trades', 'filtered_trade_signals.csv')
df = pd.read_csv(csv_path)
tickers = df['Ticker'].dropna().unique().tolist()

enriched_data = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        
        # 7-day historical data
        hist = stock.history(period="7d")
        
        # Stock fundamentals
        info = stock.info

        # Skip if no historical data is returned
        if hist.empty:
            continue

        # Calculate features
        price_change = hist["Close"].iloc[-1] - hist["Close"].iloc[0]
        volume_spike = hist["Volume"].iloc[-1] / hist["Volume"].mean()
        market_cap = info.get("marketCap", None)
        pe_ratio = info.get("trailingPE", None)
        beta = info.get("beta", None)

        enriched_data.append({
            "Ticker": ticker,
            "Price Change 7d": round(price_change, 2),
            "Volume Spike Ratio": round(volume_spike, 2),
            "Market Cap": market_cap,
            "PE Ratio": pe_ratio,
            "Beta": beta
        })

        print(f"Processed {ticker}")

        # Be nice to Yahoo Finance servers
        time.sleep(0)

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Create DataFrame
df = pd.DataFrame(enriched_data)

# Save to CSV
df.to_csv("enriched_yahoo_data.csv", index=False)

# Preview
print(df.head())




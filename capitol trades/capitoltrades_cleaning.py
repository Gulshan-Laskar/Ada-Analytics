import pandas as pd
import re
from datetime import datetime, timedelta
import requests
import time

def fetch_tickers_from_api(issuers, api_key):
    headers = {
        'Content-Type': 'text/json',
        'X-OPENFIGI-APIKEY': api_key
    }

    results = {}

    for issuer in issuers:
        query = [{"idType": "ID_NAME", "idValue": issuer}]
        try:
            response = requests.post('https://api.openfigi.com/v3/mapping', headers=headers, json=query, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data[0] and data[0]['data']:
                    ticker = data[0]['data'][0].get('ticker')
                    results[issuer] = ticker
                else:
                    results[issuer] = None
            else:
                print(f"Error {response.status_code} for {issuer}")
                results[issuer] = None
        except:
            print(f"Request failed for {issuer}")
            results[issuer] = None
        time.sleep(1)  # be polite to API

    return results

def clean_capitol_trades(file_path, output_path):
    # Load data
    df = pd.read_csv(file_path)

    # Enhanced ticker extraction
    def extract_ticker(issuer):
        if pd.isna(issuer):
            return None
        issuer = str(issuer).strip()

        # If ends with N/A → no valid ticker
        if issuer.endswith('N/A'):
            return None

        # If contains $, extract after $
        if '$' in issuer:
            parts = issuer.split('$')
            if len(parts) > 1:
                return parts[-1].strip()

        # If contains :, extract before colon
        match = re.search(r'([A-Z]+):[A-Z]+$', issuer)
        if match:
            return match.group(1)

        return None

    df['Ticker'] = df['Issuer'].apply(extract_ticker)

    missing_issuers = df.loc[df['Ticker'].isna(), 'Issuer'].dropna().unique()
    api_key = '8dd35fce-dce2-476f-9af5-e6abdbd452e4'

    ticker_map = fetch_tickers_from_api(missing_issuers, api_key)

    df['Ticker'] = df.apply(lambda x: ticker_map.get(x['Issuer'], x['Ticker']), axis=1)

    # 1️⃣ Normalize Type to lowercase
    df['Type'] = df['Type'].str.strip().str.lower()

    # 2️⃣ Clean and standardize dates directly in existing columns
    def clean_date(date_str):
        if pd.isna(date_str):
            return None
        date_str = str(date_str).strip()
        if 'Yesterday' in date_str:
            return (datetime.today() - timedelta(days=1)).date()
        if 'Today' in date_str:
            return datetime.today().date()
        return date_str  # Assume standardized date, let pandas handle it

    df['Published'] = df['Published'].apply(clean_date)
    df['Published'] = pd.to_datetime(df['Published'], errors='coerce')

    # 3️⃣ Remove rows missing Price
    df = df.dropna(subset=['Price'])

    # 4️⃣ Parse Size including millions
    def parse_size(size_str):
        if pd.isna(size_str):
            return (0, 0)
        size_str = size_str.replace('$', '').replace(',', '').strip()
        size_str = size_str.replace('K', '000').replace('M', '000000')
        size_str = size_str.replace('< ', '0–')  # Handle < 1K as 0–1000
        parts = size_str.split('–')
        if len(parts) == 2:
            try:
                lower = int(parts[0].strip())
                upper = int(parts[1].strip())
                return (lower, upper)
            except:
                return (0, 0)
        return (0, 0)

    df[['Size_Lower', 'Size_Upper']] = df['Size'].apply(lambda x: pd.Series(parse_size(x)))

    # 5️⃣ Make Size readable
    df['Size_Clean'] = df.apply(lambda x: f"${x['Size_Lower']:,} - ${x['Size_Upper']:,}" if x['Size_Lower'] > 0 else 'Unknown', axis=1)

    # Save cleaned file
    df.to_csv(output_path, index=False)

    return df

# Example usage
cleaned_df = clean_capitol_trades(
    r'capitol_trades_data.csv',
    r'cleaned_capitol_trades.csv'
)

import pandas as pd
from pathlib import Path
# Using VADER for sentiment analysis as it's simple and doesn't require model downloads
# For more advanced analysis, you could swap this with FinBERT
# You may need to run: pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
ENRICHED_INPUT_PATH = BASE_DIR / "data" / "enriched_trades.csv"
YAHOO_INPUT_PATH = BASE_DIR / "scrapers" / "yahoo_finance_data.csv"
REDDIT_INPUT_PATH = BASE_DIR / "scrapers" / "reddit_posts.csv"
OUTPUT_PATH = BASE_DIR / "data" / "features_dataset.csv"

def analyze_reddit_sentiment(reddit_df):
    """
    Calculates sentiment scores for Reddit posts and aggregates them by ticker and date.
    """
    if reddit_df.empty:
        print("Reddit data is empty, skipping sentiment analysis.")
        return pd.DataFrame(columns=['ticker', 'date', 'sentiment_score', 'mention_count'])
        
    analyzer = SentimentIntensityAnalyzer()
    
    # Ensure title is a string to prevent errors
    reddit_df['title'] = reddit_df['title'].astype(str)
    
    # Calculate sentiment for each post (using the 'compound' score from VADER)
    tqdm.pandas(desc="Analyzing Sentiment")
    reddit_df['sentiment_score'] = reddit_df['title'].progress_apply(lambda title: analyzer.polarity_scores(title)['compound'])
    
    # Convert UTC timestamp to date
    reddit_df['date'] = pd.to_datetime(reddit_df['created_utc'], unit='s').dt.date

    # Aggregate by ticker and date
    sentiment_agg = reddit_df.groupby(['ticker', 'date']).agg(
        sentiment_score=('sentiment_score', 'mean'),
        mention_count=('url', 'count')
    ).reset_index()
    
    return sentiment_agg

def calculate_volume_spikes(trades_df, yahoo_df):
    """
    Calculates trading volume spikes by comparing trade day volume to a 30-day rolling average.
    """
    if yahoo_df.empty:
        print("Yahoo data is empty, skipping volume spike calculation.")
        trades_df['volume_spike_ratio'] = None
        return trades_df

    # The yahoo data has multi-level columns (Ticker, Metric). We need to stack it.
    yahoo_df.columns = pd.MultiIndex.from_tuples(yahoo_df.columns)
    volume_df = yahoo_df.stack(level=0, future_stack=True)[['Volume']].reset_index()
    volume_df.columns = ['date', 'ticker', 'volume']
    volume_df['date'] = pd.to_datetime(volume_df['date']).dt.date

    # Calculate 30-day rolling average volume
    volume_df.sort_values(by=['ticker', 'date'], inplace=True)
    volume_df['avg_volume_30d'] = volume_df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=30, min_periods=5).mean())
    
    # Calculate the spike ratio
    volume_df['volume_spike_ratio'] = volume_df.apply(
        lambda row: row['volume'] / row['avg_volume_30d'] if pd.notnull(row['avg_volume_30d']) and row['avg_volume_30d'] > 0 else None,
        axis=1
    )

    # Merge this back into the trades dataframe
    trades_df['traded_dt_date_only'] = pd.to_datetime(trades_df['traded_dt']).dt.date
    trades_df = trades_df.merge(volume_df[['date', 'ticker', 'volume_spike_ratio']], 
                                left_on=['traded_dt_date_only', 'ticker'], 
                                right_on=['date', 'ticker'], 
                                how='left')
    trades_df.drop(['date', 'traded_dt_date_only'], axis=1, inplace=True)
    return trades_df

def main():
    """
    Takes enriched data and raw source data to engineer predictive features.
    """
    print("--- Starting Step 2: Feature Engineering ---")
    # --- Load Data ---
    if not ENRICHED_INPUT_PATH.exists():
        print(f"ERROR: Enriched data not found at {ENRICHED_INPUT_PATH}. Run step 1 first.")
        return
    df = pd.read_csv(ENRICHED_INPUT_PATH)
    df['traded_dt'] = pd.to_datetime(df['traded_dt'])


    # Load optional data sources, creating empty dataframes if they don't exist
    reddit_df = pd.read_csv(REDDIT_INPUT_PATH) if REDDIT_INPUT_PATH.exists() else pd.DataFrame()
    yahoo_df = pd.read_csv(YAHOO_INPUT_PATH, header=[0, 1], index_col=0) if YAHOO_INPUT_PATH.exists() else pd.DataFrame()

    # --- 1. Reddit Sentiment and Mentions ---
    print("Analyzing Reddit sentiment...")
    sentiment_features = analyze_reddit_sentiment(reddit_df)
    if not sentiment_features.empty:
        # Ensure date columns are of the same type for merging
        df['traded_dt_date_only'] = df['traded_dt'].dt.date
        sentiment_features['date'] = pd.to_datetime(sentiment_features['date']).dt.date
        df = df.merge(sentiment_features, left_on=['ticker', 'traded_dt_date_only'], right_on=['ticker', 'date'], how='left')
        df.drop(['date', 'traded_dt_date_only'], axis=1, inplace=True)
        print("Merged Reddit features.")
    else:
        df['sentiment_score'] = None
        df['mention_count'] = None

    # --- 2. Trading Volume Spikes ---
    print("Calculating trading volume spikes...")
    df = calculate_volume_spikes(df, yahoo_df)
    print("Merged volume spike features.")

    # --- 3. Consensus Signal ---
    print("Calculating consensus signals...")
    # CRITICAL FIX: Sort values by ticker and date BEFORE setting the index.
    # This ensures the date index is monotonic within each ticker group.
    df.sort_values(by=['ticker', 'traded_dt'], inplace=True)
    
    # Now, set the date as the index.
    df.set_index('traded_dt', inplace=True)
    
    # Group by ticker, then apply the rolling count. This will now work correctly.
    # We use .count() on a column that is NOT the grouping key. 'politician_name' is a reliable choice.
    consensus_counts = df.groupby('ticker').rolling('7D')['politician_name'].count()
    
    # The result is a series with a multi-index (ticker, traded_dt).
    # We rename it and merge it back to the main DataFrame.
    consensus_counts.name = 'consensus_score_7d'
    df = df.merge(consensus_counts, on=['ticker', 'traded_dt'], how='left')
    
    # The count includes the trade itself, so subtract 1
    df['consensus_score_7d'] = df['consensus_score_7d'] - 1
    
    # Reset the index to bring 'traded_dt' back as a column
    df.reset_index(inplace=True)
    print("Calculated 7-day consensus score.")

    # --- Save Output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"--- Feature engineering complete. Data saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()

# import pandas as pd
# import praw
# import time
# from pathlib import Path

# # --- Configuration ---
# # Base directory for robust path creation
# BASE_DIR = Path(__file__).resolve().parent
# # Input file path for cleaned Capitol Trades data
# INPUT_PATH = BASE_DIR.parent / "capitol trades new" / "capitol_trades_clean.csv"
# # Output file path for Reddit posts (our cache)
# OUTPUT_PATH = BASE_DIR / "reddit_posts.csv"

# # Reddit API credentials
# # IMPORTANT: Replace with your actual Reddit API credentials
# CLIENT_ID = "Lk7amzjYlHw4NZN4jVeMOA"
# CLIENT_SECRET = "4Xv7mFZcXHbcvUkQG98tMau-BvEFFg"
# USER_AGENT = "WSB_Sentiment_Bot/1.0 by Defiant-Fee-533"

# def load_tickers(file_path):
#     """
#     Loads unique tickers from the cleaned Capitol Trades data.
#     """
#     if not file_path.exists():
#         print(f"Error: Input file not found at {file_path}")
#         return []
#     df = pd.read_csv(file_path)
#     return df['ticker'].dropna().unique().tolist()

# def get_reddit_posts(tickers, subreddits, limit):
#     """
#     Fetches the most recent Reddit posts for a list of tickers.
#     """
#     if not tickers:
#         print("No tickers provided to fetch posts for.")
#         return pd.DataFrame()

#     if CLIENT_ID == "your_client_id" or CLIENT_SECRET == "your_client_secret":
#         print("Error: Please update your Reddit API credentials in the script.")
#         return pd.DataFrame()

#     reddit = praw.Reddit(
#         client_id=CLIENT_ID,
#         client_secret=CLIENT_SECRET,
#         user_agent=USER_AGENT,
#     )

#     all_posts = []
#     for ticker in tickers:
#         print(f"Fetching latest posts for ticker: {ticker}...")
#         try:
#             for subreddit_name in subreddits:
#                 subreddit = reddit.subreddit(subreddit_name)
#                 # Search for the newest posts
#                 for post in subreddit.search(f"title:{ticker} OR selftext:{ticker}", sort='new', limit=limit):
#                     all_posts.append({
#                         'ticker': ticker,
#                         'subreddit': subreddit_name,
#                         'title': post.title,
#                         'selftext': post.selftext,
#                         'score': post.score,
#                         'num_comments': post.num_comments,
#                         'created_utc': post.created_utc,
#                         'url': post.url,
#                     })
#             time.sleep(1) # Rate limiting
#         except Exception as e:
#             print(f"An error occurred while fetching posts for {ticker}: {e}")
#             continue

#     return pd.DataFrame(all_posts)

# def main():
#     """
#     Main function to orchestrate the Reddit data scraping process.
#     It fetches the latest posts for all required tickers and merges them with the cache.
#     """
#     required_tickers = load_tickers(INPUT_PATH)
#     if not required_tickers:
#         print("No tickers to process. Exiting.")
#         return

#     # --- Fetch latest posts for ALL required tickers ---
#     subreddits_to_scrape = ['wallstreetbets', 'stocks', 'investing']
#     post_limit = 100 # Fetch up to 100 of the newest posts for each ticker
#     latest_posts_df = get_reddit_posts(required_tickers, subreddits_to_scrape, post_limit)

#     # --- Load cache and merge ---
#     cached_df = pd.DataFrame()
#     if OUTPUT_PATH.exists():
#         print(f"Loading cached Reddit data from {OUTPUT_PATH}...")
#         cached_df = pd.read_csv(OUTPUT_PATH)
#         print(f"Cache contains {len(cached_df)} posts.")

#     if not latest_posts_df.empty:
#         # Combine old and new data, then drop duplicates, keeping the most recent record
#         combined_df = pd.concat([cached_df, latest_posts_df], ignore_index=True)
#         # The 'url' of a post is a unique identifier
#         combined_df.drop_duplicates(subset=['url'], keep='last', inplace=True)
#         print(f"Merged new posts. Total posts now: {len(combined_df)}")
#     else:
#         combined_df = cached_df
#         print("No new posts were fetched.")

#     if combined_df.empty:
#         print("No data to save.")
#         return

#     # --- Save the updated cache ---
#     OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
#     combined_df.to_csv(OUTPUT_PATH, index=False)
#     print(f"Updated Reddit data saved to {OUTPUT_PATH}")

# if __name__ == "__main__":
#     main()
















import pandas as pd
import praw
import time
from pathlib import Path
import re

# --- Configuration ---
# Base directory for robust path creation
BASE_DIR = Path(__file__).resolve().parent
# Input file path for cleaned Capitol Trades data
INPUT_PATH = BASE_DIR.parent / "capitol trades new" / "capitol_trades_clean.csv"
# Output file path for Reddit posts (our cache)
OUTPUT_PATH = BASE_DIR / "reddit_posts.csv"

# Reddit API credentials
# IMPORTANT: Replace with your actual Reddit API credentials
CLIENT_ID = "Lk7amzjYlHw4NZN4jVeMOA"
CLIENT_SECRET = "4Xv7mFZcXHbcvUkQG98tMau-BvEFFg"
USER_AGENT = "WSB_Sentiment_Bot/1.0 by Defiant-Fee-533"

def load_tickers(file_path):
    """
    Loads unique tickers from the cleaned Capitol Trades data.
    """
    if not file_path.exists():
        print(f"Error: Input file not found at {file_path}")
        return []
    df = pd.read_csv(file_path)
    # Ensure tickers are uppercase and suitable for regex
    return [t for t in df['ticker'].dropna().unique() if re.match(r'^[A-Z]{1,5}$', t)]

def get_reddit_posts_batched(tickers, subreddits, limit_per_subreddit, after_timestamp=None, batch_size=15):
    """
    Fetches Reddit posts created after a specific timestamp using batched queries.
    """
    if not tickers:
        print("No tickers provided to fetch posts for.")
        return pd.DataFrame()

    if CLIENT_ID == "your_client_id" or CLIENT_SECRET == "your_client_secret":
        print("Error: Please update your Reddit API credentials in the script.")
        return pd.DataFrame()

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )

    all_posts = []
    ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    
    print(f"Processing {len(tickers)} tickers in {len(ticker_batches)} batches...")
    if after_timestamp:
        print(f"Searching for posts newer than timestamp: {after_timestamp}")

    for i, batch in enumerate(ticker_batches):
        print(f"--- Processing Batch {i+1}/{len(ticker_batches)} ---")
        query_string = " OR ".join(batch)
        # Build the search query, adding the timestamp condition if it exists
        search_query = f'title:({query_string}) OR selftext:({query_string})'
        if after_timestamp:
            search_query += f' AND timestamp:{int(after_timestamp)}..'

        try:
            for subreddit_name in subreddits:
                print(f"Searching in r/{subreddit_name}...")
                subreddit = reddit.subreddit(subreddit_name)
                for post in subreddit.search(search_query, sort='new', limit=limit_per_subreddit):
                    post_text = post.title + " " + post.selftext
                    found_tickers = [ticker for ticker in batch if re.search(r'\b' + ticker + r'\b', post_text, re.IGNORECASE)]
                    
                    for ticker in found_tickers:
                        all_posts.append({
                            'ticker': ticker,
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'selftext': post.selftext,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'created_utc': post.created_utc,
                            'url': post.url,
                        })
            time.sleep(1) 
        except Exception as e:
            print(f"An error occurred while processing a batch: {e}")
            continue

    return pd.DataFrame(all_posts)

def main():
    """
    Main function for efficient daily scraping.
    Only fetches posts newer than the most recent post in the cache.
    """
    required_tickers = load_tickers(INPUT_PATH)
    if not required_tickers:
        print("No tickers to process. Exiting.")
        return

    last_timestamp = None
    cached_df = pd.DataFrame()
    if OUTPUT_PATH.exists():
        print(f"Loading cached Reddit data from {OUTPUT_PATH}...")
        cached_df = pd.read_csv(OUTPUT_PATH)
        if not cached_df.empty and 'created_utc' in cached_df.columns:
            # Find the timestamp of the newest post in our cache
            last_timestamp = cached_df['created_utc'].max()
        print(f"Cache contains {len(cached_df)} posts. Newest post is from {pd.to_datetime(last_timestamp, unit='s') if last_timestamp else 'N/A'}.")

    subreddits_to_scrape = ['wallstreetbets', 'stocks', 'investing']
    post_limit_per_subreddit = 100 
    
    # Pass the last timestamp to the function
    latest_posts_df = get_reddit_posts_batched(required_tickers, subreddits_to_scrape, post_limit_per_subreddit, after_timestamp=last_timestamp)

    if not latest_posts_df.empty:
        combined_df = pd.concat([cached_df, latest_posts_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['url', 'ticker'], keep='last', inplace=True)
        print(f"Added {len(combined_df) - len(cached_df)} new, unique posts. Total posts in cache now: {len(combined_df)}")
    else:
        combined_df = cached_df
        print("No new posts were fetched.")

    if combined_df.empty:
        print("No data to save.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Updated Reddit data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

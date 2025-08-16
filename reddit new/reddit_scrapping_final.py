<<<<<<< HEAD
# pip install praw pandas re time collections

import pandas as pd
import praw
import re
from collections import Counter
import time
import csv

# === Step 1: Load custom ticker list from congressional data ===
def load_custom_tickers(file_path):
    df = pd.read_csv(file_path)
    tickers = df["Ticker"].dropna().unique().tolist()
    return [ticker.upper() for ticker in tickers]

# === Step 2: Reddit API Setup ===
reddit = praw.Reddit(
    client_id="Lk7amzjYlHw4NZN4jVeMOA",
    client_secret="4Xv7mFZcXHbcvUkQG98tMau-BvEFFg",
    user_agent="WSB_Sentiment_Bot/1.0 by Defiant-Fee-533",
    check_for_async=False
)

subreddits = ['wallstreetbets', 'stocks', 'investing']
post_limit_per_ticker = 100

# === Step 3: Fetch Reddit posts for each ticker ===
def get_reddit_posts_for_ticker(ticker, max_posts=post_limit_per_ticker):
    posts = []
    query = f'title:{ticker} OR selftext:{ticker}'

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        try:
            for post in subreddit.search(query, sort='new', limit=max_posts):
                posts.append({
                    'ticker': ticker,
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'author': str(post.author)
                })
                if len(posts) >= max_posts:
                    break
        except Exception as e:
            print(f"Error fetching posts for {ticker} in r/{subreddit_name}: {e}")
        if len(posts) >= max_posts:
            break
    return posts

# === Step 4: Main Execution ===
if __name__ == "__main__":
    # Load ticker list
    ticker_list = load_custom_tickers(r"capitol trades\filtered_trade_signals.csv")
    print(f"✅ Loaded {len(ticker_list)} custom tickers: {ticker_list}")

    all_posts = []
    for ticker in ticker_list:
        print(f"\n🔍 Fetching Reddit posts for {ticker}...")
        posts = get_reddit_posts_for_ticker(ticker)
        all_posts.extend(posts)
        print(f"✅ Fetched {len(posts)} posts for {ticker}")
        time.sleep(1)  # Pause to respect Reddit's rate limits

    # Save to CSV
    if all_posts:
        df = pd.DataFrame(all_posts)
        df.to_csv(r"capitol trades\reddit_posts_for_congressional_tickers.csv", index=False)
    else:
=======
# pip install praw pandas re time collections

import pandas as pd
import praw
import re
from collections import Counter
import time
import csv

# === Step 1: Load custom ticker list from congressional data ===
def load_custom_tickers(file_path):
    df = pd.read_csv(file_path)
    tickers = df["Ticker"].dropna().unique().tolist()
    return [ticker.upper() for ticker in tickers]

# === Step 2: Reddit API Setup ===
reddit = praw.Reddit(
    client_id="Lk7amzjYlHw4NZN4jVeMOA",
    client_secret="4Xv7mFZcXHbcvUkQG98tMau-BvEFFg",
    user_agent="WSB_Sentiment_Bot/1.0 by Defiant-Fee-533",
    check_for_async=False
)

subreddits = ['wallstreetbets', 'stocks', 'investing']
post_limit_per_ticker = 100

# === Step 3: Fetch Reddit posts for each ticker ===
def get_reddit_posts_for_ticker(ticker, max_posts=post_limit_per_ticker):
    posts = []
    query = f'title:{ticker} OR selftext:{ticker}'

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        try:
            for post in subreddit.search(query, sort='new', limit=max_posts):
                posts.append({
                    'ticker': ticker,
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'author': str(post.author)
                })
                if len(posts) >= max_posts:
                    break
        except Exception as e:
            print(f"Error fetching posts for {ticker} in r/{subreddit_name}: {e}")
        if len(posts) >= max_posts:
            break
    return posts

# === Step 4: Main Execution ===
if __name__ == "__main__":
    # Load ticker list
    ticker_list = load_custom_tickers(r"capitol trades\filtered_trade_signals.csv")
    print(f"✅ Loaded {len(ticker_list)} custom tickers: {ticker_list}")

    all_posts = []
    for ticker in ticker_list:
        print(f"\n🔍 Fetching Reddit posts for {ticker}...")
        posts = get_reddit_posts_for_ticker(ticker)
        all_posts.extend(posts)
        print(f"✅ Fetched {len(posts)} posts for {ticker}")
        time.sleep(1)  # Pause to respect Reddit's rate limits

    # Save to CSV
    if all_posts:
        df = pd.DataFrame(all_posts)
        df.to_csv(r"capitol trades\reddit_posts_for_congressional_tickers.csv", index=False)
    else:
>>>>>>> 02d397b609aa60a3641617ce607f2ae2fdcdb463
        print("\n⚠️ No Reddit posts found for the provided tickers.")
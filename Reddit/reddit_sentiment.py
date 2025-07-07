# pip install pandas nltk
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# === Step 1: Download VADER Lexicon (only needed once) ===
nltk.download('vader_lexicon')

# === Step 2: Load Reddit data ===
input_file = "C:\\Users\\india\\Desktop\\Ada Analytics\\Code\\Congressional Trade Scraper\\reddit_posts_for_congressional_tickers.csv"
df = pd.read_csv(input_file)

# === Step 3: Combine title and selftext into one text field ===
df['full_text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')

# === Step 4: Initialize VADER and calculate sentiment score ===
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['full_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# === Step 5: Convert sentiment score to label ===
def label_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['sentiment_score'].apply(label_sentiment)

# === Step 6: Save post-level sentiment data ===
post_output_file = "C:\\Users\\india\\Desktop\\Ada Analytics\\Code\\Congressional Trade Scraper\\reddit_posts_with_vader_sentiment.csv"
df.to_csv(post_output_file, index=False)
print(f"✅ Post-level sentiment data saved to:\n{post_output_file}")

# === Step 7: Group by ticker and summarize ===
summary = df.groupby('ticker').agg(
    total_posts=('sentiment_label', 'count'),
    positive_posts=('sentiment_label', lambda x: (x == 'Positive').sum()),
    neutral_posts=('sentiment_label', lambda x: (x == 'Neutral').sum()),
    negative_posts=('sentiment_label', lambda x: (x == 'Negative').sum()),
    avg_sentiment_score=('sentiment_score', 'mean')
).reset_index()

# === Step 8: Save ticker-level sentiment summary ===
summary_output_file = "C:\\Users\\india\\Desktop\\Ada Analytics\\Code\\Congressional Trade Scraper\\reddit_sentiment_summary_by_ticker.csv"
summary.to_csv(summary_output_file, index=False)

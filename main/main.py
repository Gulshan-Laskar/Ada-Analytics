
import streamlit as st
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_script(script_path):
    try:
        result = subprocess.run([
            'python', script_path
        ], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

st.title("Ada Analytics: Data Pipeline Dashboard")

# 1. 13F Scraping
st.header("1. 13F Filings Scraping")
if st.button("Run 13F Scraping"):
    output = run_script(os.path.join(BASE_DIR, '13F', '13F_filings_webscraping-2.py'))
    st.text_area("13F Scraping Output", output, height=200)

# 2. Capitol Trades Scraping & Processing
st.header("2. Capitol Trades Scraping & Processing")
if st.button("Run Capitol Trades Scraping & Processing"):
    scrape_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_scrape.py'))
    clean_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_cleaning.py'))
    final_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_final.py'))
    signal_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_signal.py'))
    st.text_area("Capitol Trades Scrape Output", scrape_out, height=100)
    st.text_area("Capitol Trades Cleaning Output", clean_out, height=100)
    st.text_area("Capitol Trades Final Output", final_out, height=100)
    st.text_area("Capitol Trades Signal Output", signal_out, height=100)

# 3. Reddit Sentiment & Trend Analysis
st.header("3. Reddit Sentiment & Trend Analysis")
if st.button("Run Reddit Analysis"):
    reddit_out = run_script(os.path.join(BASE_DIR, 'Reddit', 'reddit_scrapping.py'))
    st.text_area("Reddit Analysis Output", reddit_out, height=200)

# 4. Yahoo Trend Analysis
st.header("4. Yahoo Trend Analysis")
if st.button("Run Yahoo Trend Analysis"):
    yahoo_out = run_script(os.path.join(BASE_DIR, 'yahoo', 'Yfinance-web-scrape.py'))
    st.text_area("Yahoo Trend Analysis Output", yahoo_out, height=200)

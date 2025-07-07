import streamlit as st
import subprocess
import os
import schedule
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

python_exe_venv = r'C:\Users\pusap\OneDrive\Desktop\Ada-Analytics\venv\Scripts\python.exe'
if not os.path.exists(python_exe_venv):
    python_exe_venv = 'python'

def run_script(script_path):
    try:
        result = subprocess.run([
            python_exe_venv, script_path
        ], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

st.title("Ada Analytics: Data Pipeline Dashboard")

# 1. 13F Filings Scraping
st.header("1. 13F Filings Scraping")
# Removed st.button, automatically run or show existing output
if os.path.exists(r"13F\13F_filings.csv"):
    thirteenf_df = pd.read_csv(r"13F\13F_filings.csv")
    st.dataframe(thirteenf_df)
else:
    output_13f = run_script(os.path.join(BASE_DIR, '13F', '13F_filings_webscraping-2.py'))
    st.text_area("13F Scraping Output", output_13f, height=200)
    with open("13f_output.csv", "w") as f:
        f.write(output_13f)
    thirteenf_out_df = pd.read_csv("13f_output.csv")
    st.dataframe(thirteenf_out_df)

# 2. Capitol Trades Scraping & Processing
st.header("2. Capitol Trades Scraping & Processing")
# Removed st.button, automatically run or show existing output
def run_capitol_trades():
    scrape_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_scrape.py'))
    clean_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_cleaning.py'))
    final_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_final.py'))
    signal_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_signal.py'))
    return f"{scrape_out}\n{clean_out}\n{final_out}\n{signal_out}"

if os.path.exists(r"capitol trades\trades_with_returns.csv"):
    trades_df = pd.read_csv(r"capitol trades\trades_with_returns.csv")
    st.dataframe(trades_df)
else:
    trades_result = run_capitol_trades()
    st.text_area("Capitol Trades Output", trades_result, height=400)
    with open("capitol_trades_data.csv", "w") as f:
        f.write(trades_result)
    trades_df2 = pd.read_csv("capitol_trades_data.csv")
    st.dataframe(trades_df2)

# 3. Reddit Sentiment & Trend Analysis
# Exclude for now

# 4. Yahoo Trend Analysis
st.header("4. Yahoo Trend Analysis")
# Removed st.button, automatically run or show existing output
def run_yahoo():
    return run_script(os.path.join(BASE_DIR, 'yahoo', 'Yfinance-web-scrape.py'))

if os.path.exists(r"yahoo\yahoo_finance_most_active.csv"):
    yahoo_df = pd.read_csv(r"yahoo\yahoo_finance_most_active.csv")
    st.dataframe(yahoo_df)
else:
    yahoo_out = run_yahoo()
    st.text_area("Yahoo Trend Analysis Output", yahoo_out, height=200)
    with open("yahoo_finance_most_active.csv", "w") as f:
        f.write(yahoo_out)
    yahoo_df2 = pd.read_csv("yahoo_finance_most_active.csv")
    st.dataframe(yahoo_df2)

# # Capitol Trades Visual Output
# st.header("Capitol Trades Visual Output")
# csv_path = os.path.join(BASE_DIR, 'capitol trades', 'trades_with_returns.csv')
# if os.path.exists(csv_path):
#     data_df = pd.read_csv(csv_path)
#     st.dataframe(data_df)
#     st.line_chart(data_df['Value'])

# Schedule tasks
def daily_tasks():
    run_capitol_trades()
    run_yahoo()

def quarterly_13f():
    os.remove("13f_output.txt")
    run_script(os.path.join(BASE_DIR, '13F', '13F_filings_webscraping-2.py'))

schedule.every().day.at("06:00").do(daily_tasks)
schedule.every(120).days.do(quarterly_13f)

# Start scheduling in a thread or main loop as appropriate
# (Might need special handling for Streamlit)

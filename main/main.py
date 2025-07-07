import streamlit as st
import subprocess
import os
import schedule
import time
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Ada Analytics Dashboard", page_icon=":bar_chart:", layout="wide")

# --- Custom CSS Styling ---
theme_choice = st.select_slider("Theme", ["Light Mode", "Dark Mode"], value="Dark Mode")

if theme_choice == "Dark Mode":
    st.markdown("""
    <style>
      /* position slider top-right */
      [data-testid="stSelectSlider"] { position: absolute; top: 10px; right: 10px; max-width: 200px; }
      /* make toolbar text visible */
      [data-testid="stToolbar"] { color: #cfcfcf; }
      /* app background & headings */
      .stApp { background-color: #141414; color: #e0e0e0; }
      h1,h2,h3 { color: #e0e0e0; font-family:'Poppins',sans-serif; }

      /* slider container width */
      [data-testid="stSelectSlider"] > div:first-child {
        max-width: 240px;
      }
      /* track thickness */
      [data-testid="stSelectSlider"] input[type="range"] {
        height: 10px!important; background: #444;
      }
      /* thumb size & style */
      [data-testid="stSelectSlider"] input[type="range"]::-webkit-slider-thumb {
        width: 28px!important; height: 28px!important;
        background: #666; border-radius:50%; position:relative;
      }
      /* thumb label showing current value */
      [data-testid="stSelectSlider"] input[type="range"]::-webkit-slider-thumb::after {
        content: attr(aria-valuetext);
        position: absolute; top:50%; left:50%;
        transform: translate(-50%,-50%);
        color: #fff; font-size:10px; font-weight:bold;
      }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
      /* position slider top-right */
      [data-testid="stSelectSlider"] { position: absolute; top: 10px; right: 10px; max-width: 200px; }
      /* make toolbar text visible */
      [data-testid="stToolbar"] { color: #333333; }
      /* background & headings */
      .stApp {
        background: linear-gradient(to top right,#f0ffff,#ffe4e1);
        color: #333333;
      }
      h1,h2,h3 { color: #2e3d49; font-family:'Poppins',sans-serif; }
      .stDataFrame, .st-table { background:#fff; color:#333; }

      /* slider container width */
      [data-testid="stSelectSlider"] > div:first-child {
        max-width: 240px;
      }
      /* track thickness */
      [data-testid="stSelectSlider"] input[type="range"] {
        height: 10px!important; background: #ccc;
      }
      /* thumb size & style */
      [data-testid="stSelectSlider"] input[type="range"]::-webkit-slider-thumb {
        width: 28px!important; height: 28px!important;
        background: #888; border-radius:50%; position:relative;
      }
      /* thumb label showing current value */
      [data-testid="stSelectSlider"] input[type="range"]::-webkit-slider-thumb::after {
        content: attr(aria-valuetext);
        position: absolute; top:50%; left:50%;
        transform: translate(-50%,-50%);
        color: #333; font-size:10px; font-weight:bold;
      }
    </style>
    """, unsafe_allow_html=True)

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
    signal_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_signal.py'))
    final_out = run_script(os.path.join(BASE_DIR, 'capitol trades', 'capitoltrades_final.py'))
    
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

if os.path.exists(r"yahoo\enriched_yahoo_data.csv"):
    yahoo_df = pd.read_csv(r"yahoo\enriched_yahoo_data.csv")
    st.dataframe(yahoo_df)
else:
    yahoo_out = run_yahoo()
    st.text_area("Yahoo Trend Analysis Output", yahoo_out, height=200)
    with open("enriched_yahoo_data.csv", "w") as f:
        f.write(yahoo_out)
    yahoo_df2 = pd.read_csv("enriched_yahoo_data.csv")
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

schedule.every().day.at("08:00").do(daily_tasks)
schedule.every(120).days.do(quarterly_13f)

# Start scheduling in a thread or main loop as appropriate
# (Might need special handling for Streamlit)


import streamlit as st
import subprocess
import schedule
import time
import threading
from pathlib import Path

# Define robust, relative paths to the scripts
BASE_DIR = Path(__file__).resolve().parent.parent
SCRAPING_SCRIPT = BASE_DIR / "capitol trades new" / "capitoltrades_scrape.py"
CLEANING_SCRIPT = BASE_DIR / "capitol trades new" / "capitol_trades_cleaning.py"

def run_script(script_path):
    try:
        result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
        output = result.stdout.strip()
        error = result.stderr.strip()
        if error:
            return error
        return output
    except Exception as e:
        return str(e)

def scheduled_job():
    st.session_state['log'] += run_script(SCRAPING_SCRIPT) + "\n"
    st.session_state['log'] += run_script(CLEANING_SCRIPT) + "\n"

def schedule_thread():
    schedule.every().day.at("06:00").do(scheduled_job)
    while True:
        schedule.run_pending()
        time.sleep(60)

if 'log' not in st.session_state:
    st.session_state['log'] = ""

st.title("Script Scheduler Dashboard")

if st.button("Run Scraping Script Now"):
    st.session_state['log'] += run_script(SCRAPING_SCRIPT) + "\n"

if st.button("Run Cleaning Script Now"):
    st.session_state['log'] += run_script(CLEANING_SCRIPT) + "\n"

st.text_area("Logs", st.session_state['log'], height=300)

if 'scheduler_started' not in st.session_state:
    st.session_state['scheduler_started'] = False

if not st.session_state['scheduler_started']:
    threading.Thread(target=schedule_thread, daemon=True).start()
    st.session_state['scheduler_started'] = True
    st.success("Scheduler started! Scripts will run every day at 6:00 AM.")

st.info("This dashboard will run the scraping script at 6:00 AM daily, then the cleaning script.")
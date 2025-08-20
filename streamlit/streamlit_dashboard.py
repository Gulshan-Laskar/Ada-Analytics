import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess
import sys
import os
import glob

# --- Page Configuration ---
st.set_page_config(
    page_title="Ada Analytics Trading Bot",
    page_icon="🤖",
    layout="wide"
)

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCRAPERS_DIR = BASE_DIR / "scrapers"
PROCESSING_DIR = BASE_DIR / "processing"
MODELING_DIR = BASE_DIR / "modeling"
SUGGESTIONS_DIR = DATA_DIR / "daily_suggestions"

# --- Main App ---
st.title("🤖 Ada Analytics Trading Dashboard")
st.markdown("---")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # This list has been updated with your new filenames and execution order.
    pipeline_scripts = [
        # Scraping
        (SCRAPERS_DIR / "capitoltrades_scrape.py"),
        (SCRAPERS_DIR / "yahoo_scrape.py"),
        (SCRAPERS_DIR / "reddit_scrapping.py"),
        # Processing
        (PROCESSING_DIR / "enrich_data.py"),
        (PROCESSING_DIR / "engineer_features.py"),
        (PROCESSING_DIR / "assemble_master_dataset.py"),
        # Modeling & Backtesting
        (MODELING_DIR / "model_signal.py"),
        (MODELING_DIR / "backtesting.py"),
        (MODELING_DIR / "cohort_diagnostics.py"),
        (MODELING_DIR / "ml_ready_output.py"),
        (MODELING_DIR / "ml_trainer.py"),
        (MODELING_DIR / "backtest_ml_threshold.py"),
        (MODELING_DIR / "today_suggestions.py")
    ]

    if st.button("▶️ Run Full Data Pipeline", type="primary"):
        all_scripts_found = True
        for script_path in pipeline_scripts:
            if not script_path.exists():
                st.error(f"File not found: {script_path}")
                st.warning("Please ensure the script exists and the filename matches the list.")
                all_scripts_found = False
        
        if all_scripts_found:
            st.info("All script files found. Starting pipeline...")
            log_area = st.empty()
            log_messages = []

            for script_path in pipeline_scripts:
                log_messages.append(f"Running: {script_path.name}...")
                log_area.info("\n".join(log_messages))
                
                try:
                    process = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True, text=True, check=True,
                        cwd=BASE_DIR, env=os.environ
                    )
                    log_messages.append(f"✅ Success: {script_path.name}")
                    log_area.info("\n".join(log_messages))

                except subprocess.CalledProcessError as e:
                    log_messages.append(f"❌ ERROR in {script_path.name}")
                    st.error(f"Error running {script_path.name}:")
                    st.code(e.stderr)
                    st.stop()

            log_messages.append("\n🎉 Pipeline finished successfully!")
            log_area.success("\n".join(log_messages))
            st.rerun()

    st.markdown("---")
    st.header("🧹 Maintenance")
    if st.button("Clear All Data & Models"):
        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = True
        
        if st.session_state.get("confirm_delete"):
            if st.button("Are you sure?", type="primary"):
                # Clear suggestions directory
                if SUGGESTIONS_DIR.exists():
                    for f in SUGGESTIONS_DIR.glob("*.csv"):
                        os.remove(f)
                # Clear main data directory
                for f in DATA_DIR.glob("*.*"):
                    if f.is_file():
                        try: os.remove(f)
                        except OSError as e: st.warning(f"Could not remove {f}: {e}")
                st.success("All data files and models have been deleted.")
                st.session_state.confirm_delete = False
                st.rerun()

# --- Display Results ---
st.header("🎯 Latest Suggestions")

# Find the most recent suggestions file
suggestion_files = glob.glob(str(SUGGESTIONS_DIR / "suggestions_*.csv"))
if suggestion_files:
    latest_suggestion_file = max(suggestion_files, key=os.path.getctime)
    suggestions_df = pd.read_csv(latest_suggestion_file)
    
    if not suggestions_df.empty:
        st.dataframe(suggestions_df)
        
        st.download_button(
            label="Download Suggestions as CSV",
            data=suggestions_df.to_csv(index=False).encode('utf-8'),
            file_name='latest_suggestions.csv',
            mime='text/csv',
        )
    else:
        st.info("The latest suggestion file is empty. No trades met the criteria.")

else:
    st.info("No suggestion files found. Run the pipeline to generate the latest suggestions.")

st.markdown("---")
st.header("📈 Latest Backtest Results")

summary_file = DATA_DIR / "backtest_summary_ml.csv"
if summary_file.exists():
    try:
        summary_df = pd.read_csv(summary_file)
        st.subheader("ML Backtest Summary")
        st.dataframe(summary_df)
    except Exception as e:
        st.error(f"Could not load backtest summary: {e}")
else:
    st.warning("No backtest summary found.")

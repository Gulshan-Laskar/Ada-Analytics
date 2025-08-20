import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess
import sys
import os
import glob
import json

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
st.header("🎯 Latest Suggestions (Correctly Predicted Positives)")

# Use the scored_test file which contains the y_true column
suggestions_file = DATA_DIR / "scored_test.csv"
if suggestions_file.exists():
    try:
        suggestions_df = pd.read_csv(suggestions_file)
        
        # Filter for y_true == 1 (correctly predicted positive trades)
        successful_predictions = suggestions_df[suggestions_df['y_true'] == 1].copy()
        
        if not successful_predictions.empty:
            # Define columns to display, excluding published_dt and y_true for a cleaner view
            display_cols = ['ticker', 'proba_best', 'proba_logreg', 'proba_gboost']
            # Ensure the columns exist before trying to display them
            display_cols = [col for col in display_cols if col in successful_predictions.columns]
            
            st.dataframe(successful_predictions[display_cols])
            
            st.download_button(
                label="Download Suggestions as CSV",
                data=successful_predictions[display_cols].to_csv(index=False).encode('utf-8'),
                file_name='latest_suggestions.csv',
                mime='text/csv',
            )
        else:
            st.info("No successful predictions (y_true = 1) were found in the latest test set.")

    except Exception as e:
        st.error(f"An error occurred while loading suggestions: {e}")
else:
    st.info("No suggestion file found. Run the pipeline to generate the latest suggestions.")


st.markdown("---")
st.header("🤖 Model Performance")

metrics_file = DATA_DIR / "metrics.json"
if metrics_file.exists():
    try:
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        st.metric(label="Best Performing Model", value=metrics_data.get('best_model', 'N/A').upper())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Logistic Regression Metrics")
            st.json(metrics_data.get('logreg', {}))
        with col2:
            st.subheader("Gradient Boosting Metrics")
            st.json(metrics_data.get('gboost', {}))

    except Exception as e:
        st.error(f"Could not load model metrics: {e}")
else:
    st.warning("No model performance metrics found.")

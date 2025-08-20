import pandas as pd
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
SCORED_FILE = BASE_DIR / "data" / "scored_test.csv"
OUTPUT_DIR = BASE_DIR / "data" / "daily_suggestions"
PROBA_THRESHOLD = 0.70  # Adjust if needed

def main():
    """
    Filters the latest model predictions to find actionable trade
    suggestions for the most recent date in the test set.
    """
    print("--- Generating Latest Trade Suggestions ---")
    
    if not SCORED_FILE.exists():
        print(f"ERROR: Scored predictions file not found at {SCORED_FILE}")
        print("Please run the full pipeline first to generate model predictions.")
        return

    # Load the model's predictions
    scored = pd.read_csv(SCORED_FILE, parse_dates=["published_dt"])
    
    # Automatically get the most recent date from the test data
    if scored.empty:
        print("Scored test file is empty. No suggestions to generate.")
        return
        
    target_date = scored["published_dt"].max().date()
    
    # Keep only rows for that date
    day_trades = scored[scored["published_dt"].dt.date == target_date].copy()

    # Apply the machine learning probability filter
    suggestions = day_trades[day_trades["proba_best"] >= PROBA_THRESHOLD].copy()
    
    # Sort by the highest probability
    suggestions.sort_values(by="proba_best", ascending=False, inplace=True)

    print(f"\nSuggested trades for {target_date} (proba >= {PROBA_THRESHOLD}):")
    if suggestions.empty:
        print("⚠️ None found — either no disclosures for the latest date or all were below the probability threshold.")
    else:
        # --- FIX: Include 'y_true' in the output ---
        output_cols = ['ticker', 'proba_best', 'published_dt', 'y_true']
        output_cols = [col for col in output_cols if col in suggestions.columns]

        print(suggestions[output_cols].to_string(index=False))
        print("\nTotal suggestions:", len(suggestions))

        # Optional: save to file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"suggestions_{target_date}.csv"
        suggestions[output_cols].to_csv(output_path, index=False)
        print(f"Suggestions saved to: {output_path}")

if __name__ == "__main__":
    main()

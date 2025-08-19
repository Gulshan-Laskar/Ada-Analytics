import pandas as pd
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_INPUT_PATH = BASE_DIR / "data" / "features_dataset.csv"
MASTER_OUTPUT_PATH = BASE_DIR / "data" / "master_dataset.csv"

def main():
    """
    Assembles the final master dataset. This step is for final cleaning,
    column selection, and ensuring the dataset is ready for modeling.
    """
    print("--- Starting Step 3: Assembling Master Dataset ---")
    if not FEATURES_INPUT_PATH.exists():
        print(f"ERROR: Features dataset not found at {FEATURES_INPUT_PATH}. Run step 2 first.")
        return

    df = pd.read_csv(FEATURES_INPUT_PATH)

    # --- Final Data Type Checks and Cleaning ---
    # For modeling, it's often best to fill missing values (NaN) with a neutral value like 0.
    # This indicates the absence of a signal for that feature.
    feature_columns = [
        'sentiment_score', 
        'mention_count', 
        'volume_spike_ratio', 
        'consensus_score_7d'
    ]
    
    for col in feature_columns:
        if col in df.columns:
            # This is the corrected, more robust way to fill missing values
            df[col] = df[col].fillna(0)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Performed final cleaning on feature columns.")

    # You could also select and reorder the columns you want in your final model here.
    # For now, we will keep all of them.

    # --- Save Master Dataset ---
    MASTER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_OUTPUT_PATH, index=False)
    print(f"--- Master dataset assembled. Final output saved to {MASTER_OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()

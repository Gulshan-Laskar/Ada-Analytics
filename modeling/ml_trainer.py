import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "ml_dataset.csv"
METRICS_OUT = BASE_DIR / "data" / "metrics.json"
SCORED_OUT = BASE_DIR / "data" / "scored_test.csv"
LR_MODEL_OUT = BASE_DIR / "data" / "lr_model.joblib"
GB_MODEL_OUT = BASE_DIR / "data" / "gb_model.joblib"
GB_FI_OUT = BASE_DIR / "data" / "gb_feature_importances.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_CATEGORIES = 50
PREFERRED_TARGETS = ["fwd_5d_ret","fwd_3d_ret","fwd_10d_ret","fwd_1d_ret"]

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    return df

def choose_target(df: pd.DataFrame) -> str:
    present = [c for c in PREFERRED_TARGETS if c in df.columns and df[c].notna().sum() >= 50]
    if not present:
        raise ValueError(f"No suitable target columns found. Expected one of: {PREFERRED_TARGETS}")
    print(f"[INFO] Using target: {present[0]}")
    return present[0]

def time_split(df: pd.DataFrame, test_size: float):
    df = df.sort_values("published_dt").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def prepare_features(df: pd.DataFrame, target_col: str):
    y = (df[target_col] > 0).astype(int)
    X = df.drop(columns=[c for c in df.columns if c.startswith('fwd_') or c in ["published_dt", "detail_url", target_col]])
    
    cat_cols = [c for c in X.select_dtypes(include=['object', 'category']).columns if X[c].nunique() < MAX_CATEGORIES]
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    
    for c in cat_cols:
        top = X[c].value_counts().nlargest(MAX_CATEGORIES).index
        X[c] = X[c].where(X[c].isin(top), "__OTHER__")
    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ], remainder='drop')

def evaluate(y_true, proba, thr=0.5):
    roc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else None
    pr = average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else None
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": roc, "pr_auc": pr, "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0), "threshold": thr,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "positives": int(y_true.sum()), "negatives": int((1 - y_true).sum()),
    }

def main():
    df = load_data(INPUT_FILE)
    target = choose_target(df)
    df.dropna(subset=[target], inplace=True)

    train_df, test_df = time_split(df, TEST_SIZE)
    X_train, y_train, num, cat = prepare_features(train_df, target)
    X_test, y_test, _, _ = prepare_features(test_df, target)

    pre = build_preprocessor(num, cat)
    lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300, random_state=RANDOM_STATE))])
    gb = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=300))])

    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate(y_test, lr_proba)

    gb.fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_metrics = evaluate(y_test, gb_proba)

    best_name, best_proba = ("gboost", gb_proba) if (gb_metrics.get("pr_auc") or 0) > (lr_metrics.get("pr_auc") or 0) else ("logreg", lr_proba)
    
    METRICS_OUT.write_text(json.dumps({"target": target, "n_train": len(train_df), "n_test": len(test_df), "features_numeric": num, "features_categorical": cat, "logreg": lr_metrics, "gboost": gb_metrics, "best_model": best_name}, indent=2))
    joblib.dump(lr, LR_MODEL_OUT)
    joblib.dump(gb, GB_MODEL_OUT)

    scored = test_df[["published_dt","ticker"]].copy()
    scored["y_true"] = y_test.values
    scored["proba_logreg"] = lr_proba
    scored["proba_gboost"] = gb_proba
    scored["proba_best"] = best_proba
    scored.to_csv(SCORED_OUT, index=False)

    try:
        ohe = gb.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
        fi = pd.DataFrame({"feature": num + ohe.get_feature_names_out(cat).tolist(), "importance": gb.named_steps['clf'].feature_importances_}).sort_values("importance", ascending=False)
        fi.to_csv(GB_FI_OUT, index=False)
        print(f"[OK] Feat imp -> {GB_FI_OUT}")
    except Exception as e:
        print(f"[WARN] Could not save feature importances: {e}")

    print(f"[OK] Metrics -> {METRICS_OUT}")
    print(f"[OK] Scored -> {SCORED_OUT}")
    print(f"[OK] Models -> {LR_MODEL_OUT}, {GB_MODEL_OUT}")

if __name__ == "__main__":
    main()

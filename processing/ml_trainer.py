# train_ml.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

INPUT_FILE = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/ml_dataset.csv"
METRICS_OUT = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/metrics.json"
SCORED_OUT = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/scored_test.csv"
LR_MODEL_OUT = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/lr_model.joblib"
GB_MODEL_OUT = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/gb_model.joblib"
GB_FI_OUT = "/Users/navyasrichinthapatla/Documents/Ada Analytics/new/Ada-Analytics/data/gb_feature_importances.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_CATEGORIES = 50

PREFERRED_TARGETS = ["fwd_5d_ret","fwd_3d_ret","fwd_10d_ret","fwd_1d_ret"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "published_dt" not in df.columns:
        raise ValueError("ml_dataset.csv must include 'published_dt'")
    df["published_dt"] = pd.to_datetime(df["published_dt"], errors="coerce")
    return df

def choose_target(df: pd.DataFrame) -> str:
    present = [c for c in PREFERRED_TARGETS if c in df.columns]
    if not present:
        raise ValueError(f"No target columns found. Expected one of: {PREFERRED_TARGETS}. Available: {df.columns.tolist()}")
    # pick the first with at least some non-null values
    for c in present:
        if df[c].notna().sum() >= 50:  # need at least some labels
            print(f"[INFO] Using target: {c}")
            return c
    # fall back to the first present even if sparse
    print(f"[WARN] Targets sparse; using {present[0]}")
    return present[0]

def time_split(df: pd.DataFrame, test_size: float):
    df = df.sort_values("published_dt").reset_index(drop=True)
    split_idx = int(np.floor(len(df) * (1 - test_size)))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def prepare_features(df: pd.DataFrame, target_col: str):
    drop_cols = {target_col, "published_dt"}
    drop_cols |= set(df.columns) & {"detail_url"}
    y = (df[target_col] > 0).astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in cat_cols:
        top = X[c].value_counts().nlargest(MAX_CATEGORIES).index
        X[c] = np.where(X[c].isin(top), X[c], "__OTHER__")
    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

def evaluate(y_true, proba, thr=0.5):
    if len(np.unique(y_true)) < 2:
        # metrics that need both classes will be NaN
        roc = pr = None
    else:
        roc = roc_auc_score(y_true, proba)
        pr  = average_precision_score(y_true, proba)
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": None if roc is None else float(roc),
        "pr_auc": None if pr is None else float(pr),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "threshold": float(thr),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
    }

def main():
    Path(".").mkdir(exist_ok=True)
    df = load_data(INPUT_FILE)
    target = choose_target(df)

    # keep only rows with label
    df = df[pd.notna(df[target])].copy()
    if df.empty:
        raise ValueError(f"No rows with non-null '{target}' in ml_dataset.csv")

    train_df, test_df = time_split(df, TEST_SIZE)
    X_train, y_train, num_cols, cat_cols = prepare_features(train_df, target)
    X_test,  y_test,  _,        _        = prepare_features(test_df,  target)

    pre = build_preprocessor(num_cols, cat_cols)

    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=300, C=1.0, solver="lbfgs"))])
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate(y_test, lr_proba)

    gb = Pipeline([("pre", pre),
                   ("clf", GradientBoostingClassifier(random_state=42,
                                                     n_estimators=300,
                                                     learning_rate=0.05,
                                                     max_depth=3,
                                                     subsample=1.0))])
    gb.fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_metrics = evaluate(y_test, gb_proba)

    # pick best by PR-AUC then ROC-AUC
    def key(m): return ((m.get("pr_auc") or 0.0), (m.get("roc_auc") or 0.0))
    best_name, best_proba, best_model, best_metrics = max(
        [("logreg", lr_proba, lr, lr_metrics),
         ("gboost", gb_proba, gb, gb_metrics)],
        key=lambda t: key(t[3])
    )

    # save
    Path(METRICS_OUT).write_text(json.dumps({
        "target": target,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features_numeric": num_cols,
        "features_categorical": cat_cols,
        "logreg": lr_metrics,
        "gboost": gb_metrics,
        "best_model": best_name
    }, indent=2))

    joblib.dump(lr, LR_MODEL_OUT)
    joblib.dump(gb, GB_MODEL_OUT)

    scored = test_df[["published_dt","ticker"]].copy()
    scored["y_true"] = (test_df[target] > 0).astype(int).values
    scored["proba_logreg"] = lr_proba
    scored["proba_gboost"] = gb_proba
    scored["proba_best"] = best_proba
    scored.to_csv(SCORED_OUT, index=False)

    # Optional: GB feature importances
    try:
        ohe = gb.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        all_names = num_cols + cat_names
        fi = pd.DataFrame({"feature": all_names[:len(gb.named_steps['clf'].feature_importances_)],
                           "importance": gb.named_steps["clf"].feature_importances_}) \
             .sort_values("importance", ascending=False)
        fi.to_csv(GB_FI_OUT, index=False)
    except Exception:
        pass

    print("[OK] metrics ->", METRICS_OUT)
    print("[OK] scored  ->", SCORED_OUT)
    print("[OK] models  ->", LR_MODEL_OUT, GB_MODEL_OUT)
    if Path(GB_FI_OUT).exists():
        print("[OK] feat imp->", GB_FI_OUT)
    print("Best model:", best_name)
    print("Best metrics:", json.dumps(best_metrics, indent=2))

if __name__ == "__main__":
    main()

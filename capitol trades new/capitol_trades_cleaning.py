# capitol_trades_cleaning.py
# Robust, forgiving cleaner for congressional trades:
# - Drop rows with missing price first
# - Then normalize politician, issuer, dates, sizes, etc.
# - Drop rows with missing criticals at the end
# - Output: capitol_trades_data_clean.csv

import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Robust, cross-platform paths
BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH  = BASE_DIR / "capitol trades new" / "capitol_trades_data.csv"
OUT_PATH = BASE_DIR / "capitol trades new" / "capitol_trades_clean.csv"

# ---------- Regex baselines
POLI_RE = re.compile(
    r"^\s*(?P<name>.*?)\s*(?P<party>Democrat|Republican|Independent)\s*(?P<chamber>House|Senate)\s*(?P<state>[A-Z]{2})\s*$"
)

# ---------- Helpers
def fix_date_gap(s: str) -> str:
    if pd.isna(s):
        return s
    return re.sub(r"(\b[A-Za-z]{3})(\d{4}\b)", r"\1 \2", str(s).strip())

def parse_dmy(s: Any) -> Optional[pd.Timestamp]:
    if pd.isna(s):
        return None
    s = fix_date_gap(str(s))
    ts = pd.to_datetime(s, format="%d %b %Y", errors="coerce")
    if pd.isna(ts):
        return None
    return ts

def parse_published(s: Any) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    if pd.isna(s):
        return (None, None)
    s = str(s).strip()
    if "Today" in s or "Yesterday" in s:
        return (None, s)
    dt = parse_dmy(s)
    return (dt, None)

def parse_filed_after(s: Any) -> float:
    if pd.isna(s):
        return np.nan
    m = re.search(r"(\d+)", str(s))
    return float(m.group(1)) if m else np.nan

def _to_number(token: str) -> float:
    token = token.strip().replace(",", "")
    m = re.match(r"(?i)^\$?\s*(<\s*)?(\d+(\.\d+)?)\s*([KM])?$", token)
    if not m:
        return np.nan
    val = float(m.group(2))
    suffix = m.group(4)
    if suffix:
        if suffix.upper() == "K":
            val *= 1_000
        elif suffix.upper() == "M":
            val *= 1_000_000
    return val

def parse_size_band(s: Any) -> Tuple[float, float]:
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).replace("–", "-").strip()
    if s.startswith("<"):
        hi = _to_number(s.replace("<", "").strip())
        return (0.0, hi)
    if "-" in s:
        lo, hi = [part.strip() for part in s.split("-", 1)]
        return (_to_number(lo), _to_number(hi))
    v = _to_number(s)
    return (v, v)

def parse_price(s: Any) -> float:
    if pd.isna(s):
        return np.nan
    s = str(s).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan

def safe_parse_politician(s: Any) -> Dict[str, Optional[str]]:
    if pd.isna(s):
        return {"name": None, "party": None, "chamber": None, "state": None}
    s = str(s).strip()
    m = POLI_RE.match(s)
    if not m:
        return {"name": s or None, "party": None, "chamber": None, "state": None}
    g = m.groupdict()
    return {
        "name": (g.get("name") or "").strip() or None,
        "party": (g.get("party") or "").strip() or None,
        "chamber": (g.get("chamber") or "").strip() or None,
        "state": (g.get("state") or "").strip() or None,
    }

def safe_parse_issuer(s: Any) -> Tuple[Optional[str], Optional[str]]:
    if pd.isna(s):
        return (None, None)
    s0 = str(s).strip()
    # Company + ticker :US
    m = re.match(r"^(?P<company>.*?)(?P<ticker>[A-Z.\-]+):[A-Z]{2}$", s0)
    if m:
        return (m.group("company").strip() or None, m.group("ticker").strip().upper() or None)
    # Company (TICKER)
    m = re.match(r"^(?P<company>.*?)\((?P<ticker>[A-Z.\-]+)\)\s*$", s0)
    if m:
        return (m.group("company").strip() or None, m.group("ticker").strip().upper() or None)
    # Company - TICKER
    m = re.match(r"^(?P<company>.*?)[\-\|/]\s*(?P<ticker>[A-Z.\-]+)\s*$", s0)
    if m:
        return (m.group("company").strip() or None, m.group("ticker").strip().upper() or None)
    # Company TICKER
    m = re.match(r"^(?P<company>.*\S)\s+(?P<ticker>[A-Z][A-Z.\-]+)\s*$", s0)
    if m:
        return (m.group("company").strip() or None, m.group("ticker").strip().upper() or None)
    # TICKER only
    m = re.match(r"^(?P<ticker>[A-Z][A-Z.\-]+)\s*$", s0)
    if m:
        return (None, m.group("ticker").strip().upper() or None)
    return (None, None)

# ---------- Main
def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")
    df = pd.read_csv(IN_PATH)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Replace blank-like strings with NaN
    df = df.replace({r"^\s*$": np.nan}, regex=True)

    # --- 1) Parse price column, drop rows where it's null
    if "price" in df.columns:
        df["price_usd"] = df["price"].map(parse_price)
    else:
        df["price_usd"] = np.nan

    before_price = len(df)
    df = df.dropna(subset=["price_usd"])
    dropped_price = before_price - len(df)

    # --- 2) Politician split
    pol_parts = df.get("politician", pd.Series([np.nan] * len(df))).map(safe_parse_politician)
    df["politician_name"]    = pol_parts.map(lambda d: d["name"])
    df["politician_party"]   = pol_parts.map(lambda d: d["party"])
    df["politician_chamber"] = pol_parts.map(lambda d: d["chamber"])
    df["politician_state"]   = pol_parts.map(lambda d: d["state"])

    # --- 3) Issuer -> company + ticker
    comp_tic = df.get("issuer", pd.Series([np.nan] * len(df))).map(safe_parse_issuer)
    df["company"] = comp_tic.map(lambda t: t[0])
    df["ticker"]  = comp_tic.map(lambda t: t[1])

    # --- 4) Dates
    df["traded_dt"] = df.get("traded", pd.Series([np.nan] * len(df))).map(parse_dmy)
    pub_parsed      = df.get("published", pd.Series([np.nan] * len(df))).map(parse_published)
    df["published_dt"]   = [p[0] for p in pub_parsed]
    df["published_note"] = [p[1] for p in pub_parsed]

    # --- 5) Filed After
    df["filed_after_days"] = df.get("filed_after", pd.Series([np.nan] * len(df))).map(parse_filed_after)

    # --- 6) Type / Owner normalization
    for col in ["type", "owner"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace({"nan": np.nan})
        else:
            df[col] = np.nan

    # --- 7) Size band parsing
    size_pairs = df.get("size", pd.Series([np.nan] * len(df))).map(parse_size_band)
    df["size_low_usd"]  = [p[0] for p in size_pairs]
    df["size_high_usd"] = [p[1] for p in size_pairs]

    # --- 8) Optional notional bounds
    df["notional_low"]  = df["size_low_usd"]  * df["price_usd"]
    df["notional_high"] = df["size_high_usd"] * df["price_usd"]

    # --- 9) Deduplicate
    key_cols = [
        "politician_name","politician_party","politician_chamber","politician_state",
        "ticker","traded_dt","type","size_low_usd","size_high_usd","owner"
    ]
    present_keys = [c for c in key_cols if c in df.columns]
    df = df.drop_duplicates(subset=present_keys, keep="first")

    # --- 10) Drop rows with missing criticals
    criticals = ["ticker","traded_dt","politician_name"]
    before = len(df)
    df = df.dropna(subset=[c for c in criticals if c in df.columns])
    dropped_crit = before - len(df)

    # --- 11) Column ordering
    preferred_front = [
        "politician_name","politician_party","politician_chamber","politician_state",
        "company","ticker","type","owner",
        "traded_dt","published_dt","published_note","filed_after_days",
        "size","size_low_usd","size_high_usd","price","price_usd",
        "notional_low","notional_high","detail_url"
    ]
    front = [c for c in preferred_front if c in df.columns]
    others = [c for c in df.columns if c not in front]
    df = df[front + others]

    # --- 12) Write output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # --- 13) Summary
    print(f"Wrote cleaned file: {OUT_PATH}")
    print("Final rows:", len(df))
    print("Columns:", len(df.columns))
    print(f"Dropped rows with missing price: {dropped_price}")
    print(f"Dropped rows with missing criticals {criticals}: {dropped_crit}")

if __name__ == "__main__":
    main()

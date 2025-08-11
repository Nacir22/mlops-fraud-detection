import argparse
import pandas as pd
from pathlib import Path

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Simple engineered features
    df["balance_delta"] = (df["newbalanceOrig"] - df["oldbalanceOrg"]).fillna(0.0)
    # One-hot encode 'type' (limit to common values)
    types = ["CASH_OUT","PAYMENT","TRANSFER","DEBIT","CASH_IN"]
    for t in types:
        df[f"type_{t.lower()}"] = (df["type"] == t).astype("int64")
    # Customer id (from nameOrig)
    df["customer_id"] = df["nameOrig"]
    # Event timestamp proxy from 'step' (hour steps, convert to datetime starting at epoch)
    df["event_timestamp"] = pd.to_datetime(df["step"], unit="h", origin="unix")
    # Select model features
    feat_cols = [
        "amount","balance_delta","type_cash_out","type_payment",
        "type_transfer","type_debit","type_cash_in","step","customer_id","event_timestamp","isFraud"
    ]
    return df[feat_cols]

def main(in_parquet: str = "data/processed/raw.parquet", out_parquet: str = "data/processed/features.parquet"):
    df = pd.read_parquet(in_parquet)
    feats = engineer(df)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_parquet, index=False)
    # Also create train/valid split files
    train = feats.sample(frac=0.8, random_state=42)
    valid = feats.drop(train.index)
    train.to_parquet("data/processed/train.parquet", index=False)
    valid.to_parquet("data/processed/valid.parquet", index=False)
    print(f"Saved features to {out_parquet}; train/valid splits saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--to_parquet", default="data/processed/features.parquet")
    ap.add_argument("--from_parquet", default="data/processed/raw.parquet")
    args = ap.parse_args()
    main(args.from_parquet, args.to_parquet)

import argparse
import pandas as pd
from pathlib import Path

def main(data_path: str, out_dir: str = "data/processed"):
    df = pd.read_csv(data_path)
    # Basic cleaning: keep only needed columns as provided
    keep = [
        "step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
        "nameDest","oldbalanceDest","newbalanceDest","isFraud"
    ]
    df = df[keep].copy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{out_dir}/raw.parquet", index=False)
    print(f"Saved {out_dir}/raw.parquet with {len(df)} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()
    main(args.data, args.out_dir)

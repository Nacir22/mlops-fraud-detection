import argparse
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

KEEP_SRC = [
    "step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
    "nameDest","oldbalanceDest","newbalanceDest","isFraud"
]

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balance_delta"] = (df["newbalanceOrig"] - df["oldbalanceOrg"]).fillna(0.0)
    types = ["CASH_OUT","PAYMENT","TRANSFER","DEBIT","CASH_IN"]
    for t in types:
        df[f"type_{t.lower()}"] = (df["type"] == t).astype("int64")
    feat_cols = [
        "amount","balance_delta","type_cash_out","type_payment",
        "type_transfer","type_debit","type_cash_in","step","isFraud"
    ]
    return df[feat_cols]

def main(in_parquet: str, out_parquet: str, batch_size: int, sample_frac: float):
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    tmp_feats = out_parquet + ".tmp"

    pf = pq.ParquetFile(in_parquet)
    writer = None
    rng = np.random.default_rng(42)
    total = 0

    for i, recbatch in enumerate(pf.iter_batches(batch_size=batch_size, columns=KEEP_SRC), start=1):
        chunk = recbatch.to_pandas()
        if 0 < sample_frac < 1.0:
            mask = rng.random(len(chunk)) < sample_frac
            chunk = chunk.loc[mask]
        if len(chunk) == 0:
            continue

        feats = engineer(chunk)
        table = pa.Table.from_pandas(feats, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(tmp_feats, table.schema)
        writer.write_table(table)
        total += len(feats)
        print(f"[features] batch {i}: +{len(feats)} (total={total})")

    if writer is not None:
        writer.close()
        # remplacement atomique
        Path(out_parquet).unlink(missing_ok=True)
        Path(tmp_feats).rename(out_parquet)

    # Split 80/20 en flux
    if total > 0:
        train_path = "data/processed/train.parquet"
        valid_path = "data/processed/valid.parquet"
        t_writer = v_writer = None
        j = 0
        pf2 = pq.ParquetFile(out_parquet)
        for b in pf2.iter_batches(batch_size=batch_size):
            pdf = b.to_pandas()
            # hash simple pour splitter sans tout charger
            key = np.arange(j, j + len(pdf))
            j += len(pdf)
            sel = (key % 10) < 8  # ~80%
            tdf, vdf = pdf.loc[sel], pdf.loc[~sel]
            if len(tdf):
                ttab = pa.Table.from_pandas(tdf, preserve_index=False)
                if t_writer is None:
                    t_writer = pq.ParquetWriter(train_path + ".tmp", ttab.schema)
                t_writer.write_table(ttab)
            if len(vdf):
                vtab = pa.Table.from_pandas(vdf, preserve_index=False)
                if v_writer is None:
                    v_writer = pq.ParquetWriter(valid_path + ".tmp", vtab.schema)
                v_writer.write_table(vtab)
        if t_writer: t_writer.close()
        if v_writer: v_writer.close()
        Path(train_path).unlink(missing_ok=True)
        Path(valid_path).unlink(missing_ok=True)
        Path(train_path + ".tmp").rename(train_path)
        Path(valid_path + ".tmp").rename(valid_path)
        print("✅ Train/Valid écrits.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_parquet", default="data/processed/raw.parquet")
    ap.add_argument("--to_parquet", default="data/processed/features.parquet")
    ap.add_argument("--batch_size", type=int, default=100_000)
    ap.add_argument("--sample_frac", type=float, default=0.2)
    args = ap.parse_args()
    main(args.from_parquet, args.to_parquet, args.batch_size, args.sample_frac)



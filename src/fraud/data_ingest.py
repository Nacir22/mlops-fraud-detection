import argparse
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os, tempfile, shutil

KEEP = [
    "step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
    "nameDest","oldbalanceDest","newbalanceDest","isFraud"
]

DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
}

def main(data_path: str, out_dir: str = "data/processed", chunksize: int = 100_000):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    final_path = os.path.join(out_dir, "raw.parquet")

    # dossier temp pour écriture atomique
    temp_dir = tempfile.mkdtemp(prefix="ingest_tmp_")
    tmp_path = os.path.join(temp_dir, "raw.parquet")

    reader = pd.read_csv(
        data_path,
        usecols=KEEP,
        dtype=DTYPES,
        engine="c",
        chunksize=chunksize
    )

    writer = None
    total = 0
    try:
        for i, chunk in enumerate(reader, start=1):
            table = pa.Table.from_pandas(chunk[KEEP], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            total += len(chunk)
            print(f"[chunk {i}] écrit: {len(chunk)} (cumul: {total})")
    finally:
        if writer is not None:
            writer.close()

    # validation rapide (évite un fichier partiel)
    _ = pq.ParquetFile(tmp_path)  # lève si fichier invalide

    # remplacement atomique
    if os.path.exists(final_path):
        os.remove(final_path)
    shutil.move(tmp_path, final_path)
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"✅ Sauvegardé {final_path} ({total} lignes)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()
    main(args.data, args.out_dir, args.chunksize)


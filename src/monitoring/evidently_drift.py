import argparse
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def main(ref_path: str, cur_path: str, out_html: str):
    ref = pd.read_parquet(ref_path)
    cur = pd.read_parquet(cur_path)
    feature_cols = [
        "amount","balance_delta","type_cash_out","type_payment",
        "type_transfer","type_debit","type_cash_in","step"
    ]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref[feature_cols], current_data=cur[feature_cols])
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(out_html)
    print(f"Saved drift report to {out_html}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cur", required=True)
    ap.add_argument("--out", default="reports/drift.html")
    args = ap.parse_args()
    main(args.ref, args.cur, args.out)

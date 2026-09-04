from pathlib import Path
from datetime import datetime
import pandas as pd
import duckdb

BASE_DIR = Path(__file__).parent / "data"
RAW_CSV = BASE_DIR / "raw" / "scms_delivery_history_dataset.csv"
BRONZE_DIR = BASE_DIR / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

print("[Phase 1] Reading real shipment data from:", RAW_CSV)
df = pd.read_csv(RAW_CSV, low_memory=False)

print("Rows found:", len(df))
print("Columns found:", list(df.columns))
print(df.head(3))

# Bronze rule: change NOTHING about the values themselves.
# Only add a timestamp showing when we captured this snapshot.
df["_bronze_loaded_at"] = datetime.now().isoformat()

bronze_parquet = BRONZE_DIR / "shipments_raw.parquet"
con = duckdb.connect()
con.register("raw_df", df)
con.execute(f"COPY raw_df TO '{bronze_parquet.as_posix()}' (FORMAT PARQUET)")
con.close()

print(f"\u2713 Bronze Parquet created at: {bronze_parquet}")
print(f"\u2713 Exact row count carried through: {len(df)}")

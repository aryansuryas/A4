from pathlib import Path
import pandas as pd
import duckdb

BASE_DIR = Path(__file__).parent / "data"
BRONZE_PARQUET = BASE_DIR / "bronze" / "shipments_raw.parquet"
SILVER_DIR = BASE_DIR / "silver"
SILVER_DIR.mkdir(parents=True, exist_ok=True)
SILVER_PARQUET = SILVER_DIR / "shipments_clean.parquet"

print("[Phase 2] Loading Bronze data...")
con = duckdb.connect()
df = con.execute(f"SELECT * FROM read_parquet('{BRONZE_PARQUET.as_posix()}')").df()

# Real rule 1: any value in these numeric columns that isn't actually
# a number (e.g. explanatory text) becomes a blank (NaN), not a guess.
numeric_cols = ["Weight (Kilograms)", "Freight Cost (USD)", "Line Item Quantity", "Line Item Value", "Unit Price"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Real rule 2: parse real date columns properly; anything unparseable becomes blank.
date_cols = ["Scheduled Delivery Date", "Delivered to Client Date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

# Real rule 3: fill missing vendor/country with an explicit label -- never leave it blank.
df["Vendor"] = df["Vendor"].fillna("UNKNOWN_VENDOR")
df["Country"] = df["Country"].fillna("UNKNOWN_COUNTRY")
df["Shipment Mode"] = df["Shipment Mode"].fillna("UNKNOWN_MODE")

# Real rule 4: calculate the ACTUAL delay in days, only where both real dates exist.
df["delay_days"] = (df["Delivered to Client Date"] - df["Scheduled Delivery Date"]).dt.days
df["is_delayed"] = df["delay_days"].apply(lambda d: 1 if pd.notna(d) and d > 0 else (0 if pd.notna(d) else None))

con.register("silver_df", df)
con.execute(f"COPY silver_df TO '{SILVER_PARQUET.as_posix()}' (FORMAT PARQUET)")
con.close()

print(f"\u2713 Silver dataset saved: {SILVER_PARQUET}")
print(f"\u2713 Rows with a usable delay figure: {df['delay_days'].notna().sum()} of {len(df)}")

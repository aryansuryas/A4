from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).parent / "data"
SILVER_PARQUET = BASE_DIR / "silver" / "shipments_clean.parquet"
GOLD_DIR = BASE_DIR / "gold"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
df = con.execute(f"SELECT * FROM read_parquet('{SILVER_PARQUET.as_posix()}')").df()

# Fill any still-missing numbers with the real median from this real dataset.
df["Weight (Kilograms)"] = df["Weight (Kilograms)"].fillna(df["Weight (Kilograms)"].median())
df["Unit Price"] = df["Unit Price"].fillna(df["Unit Price"].median())
df["Line Item Quantity"] = df["Line Item Quantity"].fillna(df["Line Item Quantity"].median())

# Turn real text categories into numbers a model can learn from.
df["vendor_encoded"] = df["Vendor"].astype("category").cat.codes
df["mode_encoded"] = df["Shipment Mode"].astype("category").cat.codes

features = ["Weight (Kilograms)", "Unit Price", "Line Item Quantity", "vendor_encoded", "mode_encoded"]

# Only real shipments with a genuine, known delay outcome can train or test the model.
labelled = df[df["is_delayed"].notna()].copy()
print(f"[Phase 3] {len(labelled)} of {len(df)} real shipments have a usable outcome to learn from.")

X = labelled[features]
y = labelled["is_delayed"].astype(int)

# Real evaluation: hold back 20% of real shipments the model never trains on.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)
print(f"[OK] Real accuracy on shipments the model NEVER saw during training: {accuracy:.1%}")

# Score every real shipment (not just the labelled ones) for the dashboard.
df["delay_probability"] = np.round(model.predict_proba(df[features])[:, 1], 3)
df["risk_category"] = pd.cut(df["delay_probability"], bins=[-0.1, 0.30, 0.70, 1.0], labels=["LOW", "MEDIUM", "HIGH"])

con.register("gold_df", df)
con.execute(f"""
    COPY (SELECT "ID", Vendor, Country, "Shipment Mode", "Weight (Kilograms)",
                 "Freight Cost (USD)", "Line Item Value", "Line Item Quantity", "Unit Price",
                 "Scheduled Delivery Date", "Delivered to Client Date",
                 delay_days, is_delayed, delay_probability, CAST(risk_category AS VARCHAR) AS risk_category
          FROM gold_df)
    TO '{(GOLD_DIR / "fact_shipment_analytics.parquet").as_posix()}' (FORMAT PARQUET)
""")
con.execute(f"""
    COPY (SELECT Vendor, COUNT(*) AS total_shipments,
                 ROUND(AVG(delay_days), 2) AS avg_delay_days,
                 ROUND(AVG(is_delayed) * 100, 2) AS delay_rate_pct
          FROM gold_df GROUP BY Vendor)
    TO '{(GOLD_DIR / "dim_vendor.parquet").as_posix()}' (FORMAT PARQUET)
""")
con.close()
print("[OK] Gold layer and real ML scoring complete.")

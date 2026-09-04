from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).parent / "data"
RAW_CSV = BASE_DIR / "raw" / "scms_delivery_history_dataset.csv"
BRONZE_PARQUET = BASE_DIR / "bronze" / "shipments_raw.parquet"
SILVER_PARQUET = BASE_DIR / "silver" / "shipments_clean.parquet"
GOLD_PARQUET = BASE_DIR / "gold" / "fact_shipment_analytics.parquet"
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

# ---- Plain CSV export of the finished Gold table ----
con.execute(f"""
    COPY (SELECT * FROM read_parquet('{GOLD_PARQUET.as_posix()}'))
    TO '{(EXPORT_DIR / "gold_final.csv").as_posix()}' (HEADER, DELIMITER ',')
""")
print(f"\u2713 Gold CSV exported: {EXPORT_DIR / 'gold_final.csv'}")

# ---- Plain CSV export of the untouched Bronze table, for comparison ----
con.execute(f"""
    COPY (SELECT * FROM read_parquet('{BRONZE_PARQUET.as_posix()}'))
    TO '{(EXPORT_DIR / "bronze_raw.csv").as_posix()}' (HEADER, DELIMITER ',')
""")
print(f"\u2713 Bronze CSV exported: {EXPORT_DIR / 'bronze_raw.csv'}")

# ---- Two-stage before/after: Bronze vs Gold ----
con.execute(f"""
    COPY (
        SELECT
            b."ID",
            b."Weight (Kilograms)"   AS weight_BEFORE_bronze,
            g."Weight (Kilograms)"   AS weight_AFTER_gold,
            b."Freight Cost (USD)"   AS freight_BEFORE_bronze,
            g."Freight Cost (USD)"   AS freight_AFTER_gold,
            b."Vendor"               AS vendor_BEFORE_bronze,
            g."Vendor"               AS vendor_AFTER_gold,
            g.delay_days,
            g.is_delayed,
            g.delay_probability,
            g.risk_category
        FROM read_parquet('{BRONZE_PARQUET.as_posix()}') b
        JOIN read_parquet('{GOLD_PARQUET.as_posix()}') g ON b."ID" = g."ID"
        ORDER BY b."ID"
    )
    TO '{(EXPORT_DIR / "bronze_vs_gold_comparison.csv").as_posix()}' (HEADER, DELIMITER ',')
""")
print(f"\u2713 2-stage comparison exported: {EXPORT_DIR / 'bronze_vs_gold_comparison.csv'}")

# ---- Four-stage trace: Raw -> Bronze -> Silver -> Gold, same shipment ID ----
# This is the one that actually shows WHERE each fix happens. Raw and Bronze
# columns should be identical (Bronze changes nothing but adds a timestamp --
# that's the point). Silver is where junk text disappears. Gold is where
# blanks get filled and risk scores appear.
con.execute(f"""
    COPY (
        SELECT
            r."ID",
            r."Weight (Kilograms)"::VARCHAR AS weight_1_RAW,
            b."Weight (Kilograms)"::VARCHAR AS weight_2_BRONZE,
            s."Weight (Kilograms)"::VARCHAR AS weight_3_SILVER,
            g."Weight (Kilograms)"::VARCHAR AS weight_4_GOLD,
            r."Freight Cost (USD)"::VARCHAR AS freight_1_RAW,
            b."Freight Cost (USD)"::VARCHAR AS freight_2_BRONZE,
            s."Freight Cost (USD)"::VARCHAR AS freight_3_SILVER,
            g."Freight Cost (USD)"::VARCHAR AS freight_4_GOLD,
            r."Vendor"                      AS vendor_1_RAW,
            b."Vendor"                      AS vendor_2_BRONZE,
            s."Vendor"                      AS vendor_3_SILVER,
            g."Vendor"                      AS vendor_4_GOLD,
            s.delay_days                    AS delay_days_appears_in_SILVER,
            s.is_delayed                    AS is_delayed_appears_in_SILVER,
            g.delay_probability             AS delay_probability_appears_in_GOLD,
            g.risk_category                 AS risk_category_appears_in_GOLD
        FROM read_csv_auto('{RAW_CSV.as_posix()}') r
        JOIN read_parquet('{BRONZE_PARQUET.as_posix()}') b ON r."ID" = b."ID"
        JOIN read_parquet('{SILVER_PARQUET.as_posix()}') s ON r."ID" = s."ID"
        JOIN read_parquet('{GOLD_PARQUET.as_posix()}') g ON r."ID" = g."ID"
        ORDER BY r."ID"
    )
    TO '{(EXPORT_DIR / "all_layers_comparison.csv").as_posix()}' (HEADER, DELIMITER ',')
""")
print(f"\u2713 4-stage trace exported: {EXPORT_DIR / 'all_layers_comparison.csv'}")

con.close()
print("\nFor teaching: open all_layers_comparison.csv. Sort by weight_1_RAW and look")
print("for a row where it says 'Weight Captured Separately'. Read across that row:")
print("RAW and BRONZE columns will match exactly (Bronze changes nothing). SILVER")
print("turns the junk text blank. GOLD fills the blank with a real median number")
print("and adds the risk score. That single row tells the whole story.")


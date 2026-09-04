import os
import re
from pathlib import Path
import duckdb
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).parent / "data"
FACT_PARQUET = BASE_DIR / "gold" / "fact_shipment_analytics.parquet"

# Load API Key from environment variable or .env file
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY and API_KEY not in ["YOUR_ACTUAL_API_KEY_HERE", "ADD_API_KEY_HERE"] else None

GEMINI_MODEL = "gemini-3.5-flash-lite"


def ask_supply_chain_llm(user_query: str):
    if not client:
        return "ERROR: No valid GEMINI_API_KEY found. Please set GEMINI_API_KEY in your .env file or environment.", None

    schema_info = f"""
    Table: read_parquet('{FACT_PARQUET.as_posix()}')
    Columns: ID, Vendor, Country, "Shipment Mode", "Weight (Kilograms)",
             "Freight Cost (USD)", "Scheduled Delivery Date", "Delivered to Client Date",
             delay_days, is_delayed, delay_probability, risk_category
    """

    prompt = f"""
    You are an expert SQL engineer for a logistics company.
    Given this DuckDB schema: {schema_info}
    User Request: "{user_query}"
    GUARDRAIL RULE: If the request is NOT about real shipments, logistics,
    vendors, weights, freight or delays, reply EXACTLY with:
    ERROR: Unrelated Query. I can only assist with logistics and supply chain data analysis.
    Otherwise, generate a valid DuckDB SQL query. Return ONLY the SQL, in ```sql ... ``` fences.
    """

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        raw_text = response.text.strip()
        if "ERROR: Unrelated Query" in raw_text:
            return raw_text, None
        match = re.search(r"```sql\s*(.*?)\s*```", raw_text, re.DOTALL)
        sql_query = match.group(1).strip() if match else raw_text
        con = duckdb.connect()
        result_df = con.execute(sql_query).df()
        con.close()
        return sql_query, result_df
    except Exception as e:
        return f"Query Failed: {str(e)}", None


def generate_rca_and_email(shipment_data: dict) -> str:
    if not client:
        return "ERROR: No real API key found. Paste your key in ai_copilot.py."
    prompt = f"""
    Write a 2-sentence Root Cause Analysis and a brief customer delay
    apology email for shipment {shipment_data.get('ID')}, which has a
    {round(float(shipment_data.get('delay_probability', 0)) * 100, 1)}%
    real ML-predicted risk of delay.
    """
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"Generation Failed: {str(e)}"

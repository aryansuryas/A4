import os
from pathlib import Path
import json
import re
import time
import duckdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).parent / "data"
GOLD_PARQUET = BASE_DIR / "gold" / "fact_shipment_analytics.parquet"
DIM_VENDOR = BASE_DIR / "gold" / "dim_vendor.parquet"
LOG_CSV = BASE_DIR / "gold" / "agent_action_log.csv"

# Load API Key from environment variable or .env file
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY and API_KEY not in ["YOUR_ACTUAL_API_KEY_HERE", "ADD_API_KEY_HERE"] else None
GEMINI_MODEL = "gemini-3.5-flash-lite"
# Free tier for this model allows 15 requests PER MINUTE. Each shipment's
# investigation can itself trigger several real calls (the model may call
# more than one tool before answering), so both limits below exist to keep
# a single run safely under that cap.
MAX_TOOL_CALLS_PER_SHIPMENT = 3
SECONDS_BETWEEN_SHIPMENTS = 5
REQUEST_TIMEOUT_SECONDS = 30  # google-genai has known bugs where its own timeout
                               # config doesn't reliably fire -- this is our own
                               # hard cutoff, enforced from outside the SDK.


# ==================== Real, shared data tools ====================

def get_high_risk_shipments_needing_action(action_type: str, limit: int = 10, exclude_mode: str = None) -> list:
    """
    Returns real shipments currently scored HIGH risk that do not yet
    have THIS SPECIFIC action logged from a previous agent run. Each
    item has: id, vendor, shipment_mode, weight_kg, delay_probability.

    action_type: which action to check for, e.g. "root_cause_investigated"
    or "routing_recommended". Kept separate per action so one agent's
    completed work doesn't block a different agent from processing the
    same real shipment.
    limit: how many real shipments to return (default 10).
    exclude_mode: optionally skip shipments already on this real mode,
    useful for reaching a mix of modes faster during testing.
    """
    con = duckdb.connect()
    already_done = set()
    if LOG_CSV.exists():
        done_df = pd.read_csv(LOG_CSV)
        done_df = done_df[done_df["action"] == action_type]
        already_done = set(done_df["shipment_id"].astype(str).tolist())

    df = con.execute(f"""
        SELECT "ID", Vendor, "Shipment Mode", "Weight (Kilograms)", delay_probability
        FROM read_parquet('{GOLD_PARQUET.as_posix()}')
        WHERE risk_category = 'HIGH'
        ORDER BY delay_probability DESC
    """).df()
    con.close()

    df = df[~df["ID"].astype(str).isin(already_done)]
    if exclude_mode:
        df = df[df["Shipment Mode"] != exclude_mode]
    results = []
    for _, row in df.head(limit).iterrows():
        results.append({
            "id": str(row["ID"]),
            "vendor": row["Vendor"],
            "shipment_mode": row["Shipment Mode"],
            "weight_kg": float(row["Weight (Kilograms)"]) if pd.notna(row["Weight (Kilograms)"]) else None,
            "delay_probability": float(row["delay_probability"]),
        })
    return results


def check_vendor_history(vendor: str) -> dict:
    """
    Returns this real vendor's historical shipment stats: total_shipments,
    avg_delay_days, delay_rate_pct. Use this to see whether a vendor is
    chronically unreliable, versus this being an unusual one-off delay.
    """
    con = duckdb.connect()
    row = con.execute(
        f"SELECT total_shipments, avg_delay_days, delay_rate_pct "
        f"FROM read_parquet('{DIM_VENDOR.as_posix()}') WHERE Vendor = ?",
        [vendor],
    ).df()
    con.close()
    if row.empty:
        return {"vendor": vendor, "found": False, "note": "No real history found for this vendor."}
    r = row.iloc[0]
    return {
        "vendor": vendor, "found": True,
        "total_shipments": int(r["total_shipments"]),
        "avg_delay_days": float(r["avg_delay_days"]),
        "delay_rate_pct": float(r["delay_rate_pct"]),
    }


def check_shipment_mode_stats() -> list:
    """
    Returns real delay statistics grouped by shipment mode (e.g. Air,
    Ocean, Truck): shipment_mode, total_shipments, delay_rate_pct,
    avg_delay_days. Use this to see whether a mode of transport is
    generally more or less reliable than others.
    """
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT "Shipment Mode" AS shipment_mode, COUNT(*) AS total_shipments,
               ROUND(AVG(is_delayed) * 100, 1) AS delay_rate_pct,
               ROUND(AVG(delay_days), 1) AS avg_delay_days
        FROM read_parquet('{GOLD_PARQUET.as_posix()}')
        WHERE is_delayed IS NOT NULL
        GROUP BY "Shipment Mode"
    """).df()
    con.close()
    return df.to_dict("records")


def check_weight_percentile(weight_kg: float) -> dict:
    """
    Returns where this real weight (in kilograms) ranks among all real
    shipments, as a percentile from 0-100, plus the real median and max
    weight for context. A high percentile means this shipment is
    unusually heavy compared to the rest of the real dataset.
    """
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT "Weight (Kilograms)" AS w FROM read_parquet('{GOLD_PARQUET.as_posix()}')
        WHERE "Weight (Kilograms)" IS NOT NULL
    """).df()
    con.close()
    if df.empty:
        return {"weight_kg": weight_kg, "percentile": None, "note": "No real weight data available."}
    percentile = float((df["w"] <= weight_kg).mean() * 100)
    return {
        "weight_kg": weight_kg,
        "percentile": round(percentile, 1),
        "real_median_weight": float(df["w"].median()),
        "real_max_weight": float(df["w"].max()),
    }


def recommend_alternative_routing(current_mode: str) -> dict:
    """
    Compares real delay rates across shipment modes and recommends a
    better-performing real alternative to current_mode, if one exists
    with enough real historical shipments (at least 10) to be reliable.
    Excludes UNKNOWN_MODE from candidate alternatives since it represents
    missing/unclassified data, not a real, actionable transport option.
    """
    stats = check_shipment_mode_stats()
    reliable = [
        s for s in stats
        if s["total_shipments"] >= 10 and s["shipment_mode"] != "UNKNOWN_MODE"
    ]
    current = next((s for s in reliable if s["shipment_mode"] == current_mode), None)
    alternatives = sorted(
        [s for s in reliable if s["shipment_mode"] != current_mode],
        key=lambda s: s["delay_rate_pct"],
    )
    if not current or not alternatives:
        return {"current_mode": current_mode, "recommended_mode": None,
                "note": "Not enough real, classified data to make a reliable routing recommendation."}
    best = alternatives[0]
    if best["delay_rate_pct"] >= current["delay_rate_pct"]:
        return {"current_mode": current_mode, "recommended_mode": None,
                "current_delay_rate_pct": current["delay_rate_pct"],
                "note": "Current mode already has the best real delay rate among real, classified alternatives."}
    return {
        "current_mode": current_mode,
        "current_delay_rate_pct": current["delay_rate_pct"],
        "recommended_mode": best["shipment_mode"],
        "recommended_delay_rate_pct": best["delay_rate_pct"],
        "expected_improvement_pct": round(current["delay_rate_pct"] - best["delay_rate_pct"], 1),
        "sample_size": best["total_shipments"],
    }


def log_action_taken(shipment_id: str, action: str) -> str:
    """
    Records that a real action was taken for a real shipment ID, so
    future runs skip it instead of repeating the same work.
    """
    row = pd.DataFrame([{"shipment_id": shipment_id, "action": action}])
    if LOG_CSV.exists():
        row.to_csv(LOG_CSV, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_CSV, mode="w", header=True, index=False)
    return f"Logged: {action} for shipment {shipment_id}"


# ==================== JSON-safe model call helper ====================

def _call_with_tools_expect_json(prompt: str, tools: list) -> dict:
    """
    Calls Gemini (single fixed model, GEMINI_MODEL) with real tools
    available, and safely extracts a JSON object from the response even
    if the model wraps it in ```json fences or adds extra text around it.
    """
    if not client:
        return {"error": "No valid GEMINI_API_KEY found. Set GEMINI_API_KEY in your .env file or environment."}
    print(f"    -> calling {GEMINI_MODEL} (timeout {REQUEST_TIMEOUT_SECONDS}s)...")
    try:
        chat = client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=MAX_TOOL_CALLS_PER_SHIPMENT
                ),
            ),
        )
        # Run the real network call in a background thread so a hung
        # request (a known, documented issue in this SDK) can never
        # freeze the whole script -- we just stop waiting and report it.
        # NOTE: not using ThreadPoolExecutor as a context manager here --
        # its __exit__ blocks until the background thread finishes, which
        # defeats the timeout entirely when that thread is genuinely hung.
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(chat.send_message, prompt)
        try:
            response = future.result(timeout=REQUEST_TIMEOUT_SECONDS)
        finally:
            executor.shutdown(wait=False)
        raw = response.text.strip()
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        json_text = match.group(1).strip() if match else raw
        print(f"    -> {GEMINI_MODEL} responded.")
        return json.loads(json_text)
    except FutureTimeoutError:
        print(f"    -> {GEMINI_MODEL} did not respond within {REQUEST_TIMEOUT_SECONDS}s.")
        return {"error": f"{GEMINI_MODEL} timed out after {REQUEST_TIMEOUT_SECONDS}s (no response -- check network/proxy/firewall)"}
    except json.JSONDecodeError:
        return {"error": "Model did not return valid JSON.", "raw_response": raw}
    except Exception as e:
        print(f"    -> {GEMINI_MODEL} failed: {e}")
        return {"error": f"{GEMINI_MODEL} failed: {e}"}


# ==================== Agent 1: Root-Cause Investigation ====================

def run_root_cause_agent(limit: int = 3) -> list:
    """
    For each real, unhandled HIGH-risk shipment (up to `limit` of them),
    investigates using real tools, determines a likely cause, and drafts
    a real mitigation email. Returns a list of structured results.
    Default limit kept low to stay within the free-tier per-minute quota.
    """
    shipments = get_high_risk_shipments_needing_action(action_type="root_cause_investigated", limit=limit)
    reports = []
    for i, s in enumerate(shipments):
        if i > 0:
            time.sleep(SECONDS_BETWEEN_SHIPMENTS)
        prompt = f"""
        Investigate this real HIGH-risk shipment using the tools available
        to you (check_vendor_history, check_shipment_mode_stats,
        check_weight_percentile). Use whichever tools are relevant.

        Shipment details:
        - ID: {s['id']}
        - Vendor: {s['vendor']}
        - Shipment Mode: {s['shipment_mode']}
        - Weight (kg): {s['weight_kg']}
        - Delay probability: {s['delay_probability']}

        Respond with ONLY a JSON object, no other text, in this exact shape:
        {{"shipment_id": "...", "likely_cause": "...", "evidence": "...",
          "email_subject": "...", "email_body": "..."}}
        """
        result = _call_with_tools_expect_json(
            prompt, [check_vendor_history, check_shipment_mode_stats, check_weight_percentile]
        )
        result.setdefault("shipment_id", s["id"])
        result["vendor"] = s["vendor"]
        reports.append(result)
        if "error" not in result:
            log_action_taken(s["id"], "root_cause_investigated")
    return reports


# ============= Agent 2: Alternative-Routing Recommendation =============

def run_routing_agent(limit: int = 3) -> list:
    """
    For each real, unhandled HIGH-risk shipment (up to `limit` of them),
    recommends a better-performing real shipment mode where the real
    data supports it. Returns a list of structured recommendations.
    Default limit kept low to stay within the free-tier per-minute quota.
    """
    shipments = get_high_risk_shipments_needing_action(action_type="routing_recommended", limit=limit, exclude_mode="Air")
    reports = []
    for i, s in enumerate(shipments):
        if i > 0:
            time.sleep(SECONDS_BETWEEN_SHIPMENTS)
        prompt = f"""
        This real shipment is HIGH risk and currently moving via
        {s['shipment_mode']}. Use check_shipment_mode_stats and
        recommend_alternative_routing to see if a real, better-performing
        alternative mode exists.

        Shipment details:
        - ID: {s['id']}
        - Vendor: {s['vendor']}
        - Shipment Mode: {s['shipment_mode']}
        - Weight (kg): {s['weight_kg']}
        - Delay probability: {s['delay_probability']}

        Respond with ONLY a JSON object, no other text, in this exact shape:
        {{"shipment_id": "...", "current_mode": "...", "recommended_mode": "... or null",
          "expected_improvement_pct": 0.0, "rationale": "...",
          "email_subject": "...", "email_body": "..."}}
        """
        result = _call_with_tools_expect_json(
            prompt, [check_shipment_mode_stats, recommend_alternative_routing]
        )
        result.setdefault("shipment_id", s["id"])
        result["vendor"] = s["vendor"]
        reports.append(result)
        if "error" not in result:
            log_action_taken(s["id"], "routing_recommended")
    return reports

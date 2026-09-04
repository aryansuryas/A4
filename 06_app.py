from pathlib import Path
from urllib.parse import quote
import duckdb
import plotly.express as px
import streamlit as st
from ai_copilot import ask_supply_chain_llm, generate_rca_and_email
from agentic_core import run_root_cause_agent, run_routing_agent

st.set_page_config(page_title="Supply Chain Control Tower", layout="wide")

BASE_DIR = Path(__file__).parent / "data"
FACT_PARQUET = BASE_DIR / "gold" / "fact_shipment_analytics.parquet"
DIM_VENDOR = BASE_DIR / "gold" / "dim_vendor.parquet"

st.title("🚚 Real Supply Chain Control Tower")


@st.cache_data
def load_gold_data():
    con = duckdb.connect()
    fact = con.execute(f"SELECT * FROM read_parquet('{FACT_PARQUET.as_posix()}')").df()
    vendor = con.execute(f"SELECT * FROM read_parquet('{DIM_VENDOR.as_posix()}')").df()
    con.close()
    return fact, vendor


fact_df, vendor_df = load_gold_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard & Stats",
    "🤖 Text-to-SQL AI Analyst",
    "⚠️ High-Risk Incident Mitigation",
    "🕵️ Agentic AI Control Room",
])

# ===================== TAB 1: DASHBOARD & STATS =====================
with tab1:
    st.subheader("Real shipment overview")

    total_shipments = len(fact_df)
    known = fact_df["is_delayed"].notna().sum()
    delay_rate = fact_df["is_delayed"].mean() * 100 if known else 0
    avg_delay = fact_df.loc[fact_df["delay_days"].notna(), "delay_days"].mean()
    high_risk_count = (fact_df["risk_category"] == "HIGH").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total real shipments", f"{total_shipments:,}")
    col2.metric("Real delay rate", f"{delay_rate:.1f}%")
    col3.metric("Avg real delay (days)", f"{avg_delay:.1f}" if avg_delay == avg_delay else "n/a")
    col4.metric("HIGH-risk shipments", f"{high_risk_count:,}")

    left, right = st.columns(2)

    with left:
        st.markdown("**Risk category breakdown**")
        risk_counts = fact_df["risk_category"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
        fig1 = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            labels={"x": "Risk category", "y": "Number of shipments"},
            color=risk_counts.index,
            color_discrete_map={"LOW": "#00C6C2", "MEDIUM": "#F5A623", "HIGH": "#FF0000"},
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        st.markdown("**Top 10 vendors by real delay rate**")
        top_vendors = vendor_df.sort_values("delay_rate_pct", ascending=False).head(10)
        fig2 = px.bar(
            top_vendors, x="delay_rate_pct", y="Vendor", orientation="h",
            labels={"delay_rate_pct": "Delay rate (%)", "Vendor": ""},
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Weight vs. predicted delay probability (real scored shipments)**")
    sample = fact_df.dropna(subset=["Weight (Kilograms)", "delay_probability"]).sample(
        min(1000, len(fact_df)), random_state=42
    )
    fig3 = px.scatter(
        sample, x="Weight (Kilograms)", y="delay_probability", color="risk_category",
        color_discrete_map={"LOW": "#00C6C2", "MEDIUM": "#F5A623", "HIGH": "#FF0000"},
        labels={"delay_probability": "Predicted delay probability"},
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Vendor summary table (real, from dim_vendor)**")
    st.dataframe(vendor_df.sort_values("total_shipments", ascending=False), use_container_width=True)

# ===================== TAB 2: AI ANALYST =====================
with tab2:
    st.subheader("Query the real lakehouse in plain English")

    sample_queries = [
        "Which vendor has the highest number of HIGH risk shipments?",
        "List the total shipments and average delay days by vendor.",
        "Show the top 5 heaviest shipments that were delayed.",
        "What is the average weight for HIGH risk versus LOW risk shipments?",
        "How many shipments have a negative weight?",
    ]
    selection = st.selectbox("Choose a pre-set question or select Custom:", ["Custom Query"] + sample_queries)
    user_prompt = st.text_input(
        "Enter your analytical question:",
        selection if selection != "Custom Query" else "",
    )

    if st.button("Execute AI Query"):
        if not user_prompt:
            st.warning("Please enter a question.")
        else:
            with st.spinner("AI is generating and executing DuckDB SQL..."):
                sql_code, result_data = ask_supply_chain_llm(user_prompt)
            if result_data is None and "ERROR" in sql_code:
                st.error(sql_code)
            else:
                st.code(sql_code, language="sql")
                st.dataframe(result_data, use_container_width=True)

# ===================== TAB 3: HIGH-RISK MITIGATION =====================
with tab3:
    st.subheader("Autonomous RCA generator")
    high_risk = fact_df[fact_df["risk_category"] == "HIGH"]
    if high_risk.empty:
        st.info("No real shipments are currently scored HIGH risk.")
    else:
        selected_id = st.selectbox("Select a real HIGH-risk shipment ID:", high_risk["ID"].tolist())
        if st.button("Generate RCA Email"):
            with st.spinner("Analyzing real risk factors..."):
                row = high_risk[high_risk["ID"] == selected_id].iloc[0].to_dict()
                st.markdown(generate_rca_and_email(row))

# ===================== TAB 4: AGENTIC AI CONTROL ROOM =====================
with tab4:
    st.subheader("Agentic AI Control Room")
    st.caption(
        "Each button gives the agent one real goal. It decides on its own which "
        "real tools to call and how many shipments to act on — the steps below "
        "aren't scripted by a human."
    )

    def render_agent_reports(reports, kind):
        if not reports:
            st.info("No unhandled real HIGH-risk shipments were found.")
            return
        for r in reports:
            if "error" in r:
                st.error(f"Shipment {r.get('shipment_id', '?')}: {r['error']}")
                continue

            header = f"Shipment {r.get('shipment_id')} — {r.get('vendor', '')}"
            with st.expander(header):
                if kind == "root_cause":
                    st.markdown(f"**Likely cause:** {r.get('likely_cause', 'n/a')}")
                    st.markdown(f"**Evidence:** {r.get('evidence', 'n/a')}")
                else:
                    rec_mode = r.get("recommended_mode") or "No change recommended"
                    st.markdown(f"**Current mode:** {r.get('current_mode', 'n/a')}")
                    st.markdown(f"**Recommended mode:** {rec_mode}")
                    st.markdown(f"**Expected improvement:** {r.get('expected_improvement_pct', 'n/a')}%")
                    st.markdown(f"**Rationale:** {r.get('rationale', 'n/a')}")

                subject = r.get("email_subject", "")
                body = r.get("email_body", "")
                if subject or body:
                    st.markdown("---")
                    st.markdown("**Drafted email** (no vendor address is stored in this real "
                                "dataset, so add the real recipient yourself before sending):")
                    mailto = f"mailto:?subject={quote(subject)}&body={quote(body)}"
                    st.markdown(f"[📧 Open this draft in your email client]({mailto})")
                    st.code(f"Subject: {subject}\n\n{body}", language=None)

    col_a, col_b = st.columns(2)
    run_root_cause = col_a.button("🔍 Run Root-Cause Investigation")
    run_routing = col_b.button("🔀 Run Alternative-Routing Recommendation")

    if run_root_cause:
        with st.spinner("Agent is investigating real HIGH-risk shipments..."):
            reports = run_root_cause_agent()
        render_agent_reports(reports, "root_cause")

    if run_routing:
        with st.spinner("Agent is evaluating real routing alternatives..."):
            reports = run_routing_agent()
        render_agent_reports(reports, "routing")

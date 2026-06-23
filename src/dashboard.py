"""
AI Pipeline Monitoring Dashboard (Streamlit)

Reads data/pipeline.db in real-time and displays:
  - KPI Dashboard (total processed / successful assets / DLQ failures / avg retries)
  - Dead Letter Cemetery (red error panel)
  - Self-Correction Audit Trail (per-doc_id drill-down)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline.db"


# ──────────────────────────────────────────────
#  DB helpers
# ──────────────────────────────────────────────
@st.cache_resource
def get_connection() -> sqlite3.Connection:
    """Return a cached SQLite connection."""
    if not DB_PATH.exists():
        st.error(f"❌ Database not found at {DB_PATH}")
        st.stop()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str) -> pd.DataFrame:
    """Execute SQL and return results as a DataFrame."""
    conn = get_connection()
    return pd.read_sql_query(sql, conn)


# ──────────────────────────────────────────────
#  Layout
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Financial AI Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Dark-mode friendly custom CSS ──
st.markdown(
    """
<style>
.card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.5rem 1rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,.4);
}
.card .value {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
}
.card .label {
    font-size: 0.85rem;
    color: #a0a0b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.pass  .value { color: #00e676; }
.fail  .value { color: #ff5252; }
.total .value { color: #40c4ff; }
.retry .value { color: #ffd740; }
h1, h2, h3 { font-weight: 600; }
.section-title {
    border-bottom: 2px solid #2a2a4a;
    padding-bottom: 0.4rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
#  Data
# ──────────────────────────────────────────────
df = query("SELECT * FROM extractions")
df_dlq = query("SELECT * FROM dead_letter_queue")
df_corr = query("SELECT * FROM correction_logs")

total_files = len(df)
dlq_count = len(df_dlq)

# Determine success count
if "validation_passed" in df.columns:
    success_count = len(df[df["validation_passed"] == 1])
else:
    success_count = len(df[df["is_current"] == 1])

avg_retry = round(df["retry_count"].mean(), 2) if len(df) else 0.0

# ──────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────
st.title("📊 Financial AI Pipeline – Monitoring Dashboard")
st.caption(f"Data source: `{DB_PATH}` · Last refreshed: on page load")

# ──────────────────────────────────────────────
#  1. KPI Dashboard
# ──────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Core KPIs</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f'<div class="card total"><p class="value">{total_files}</p>'
        f'<p class="label">📂 Total Processed</p></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f'<div class="card pass"><p class="value">{success_count}</p>'
        f'<p class="label">✅ Successful Assets</p></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f'<div class="card fail"><p class="value">{dlq_count}</p>'
        f'<p class="label">☠️ Dead Letter Queue</p></div>',
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f'<div class="card retry"><p class="value">{avg_retry}</p>'
        f'<p class="label">🔄 Avg Retry Count</p></div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
#  2. Version History (Plotly)
# ──────────────────────────────────────────────
st.markdown(
    '<div class="section-title">📈 Asset Version History</div>', unsafe_allow_html=True
)

if len(df):
    fig = px.bar(
        df,
        x="doc_id",
        y="version",
        color="is_current",
        hover_data={"file_path": True, "doc_type": True, "retry_count": True},
        labels={
            "doc_id": "Document ID",
            "version": "Version",
            "is_current": "Is Active",
            "file_path": "Source File",
        },
        color_discrete_map={0: "#ff5252", 1: "#00e676"},
        title="Version Evolution per Document (Red=Archived / Green=Active)",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("📋 View extractions raw data"):
        st.dataframe(df, width="stretch", height=300)
else:
    st.info("ℹ️ No data in extractions table yet")

# ──────────────────────────────────────────────
#  3. Dead Letter Cemetery (Error Handling)
# ──────────────────────────────────────────────
st.markdown(
    '<div class="section-title">☠️ Dead Letter Cemetery — Error Handling</div>',
    unsafe_allow_html=True,
)

if df_dlq.empty:
    st.success("🎉 Dead Letter Queue is empty! All documents processed successfully.")
else:
    st.error(f"⚠️ {len(df_dlq)} document(s) permanently failed and entered the DLQ:")
    for _, row in df_dlq.iterrows():
        with st.expander(f"💥 {Path(row.file_path).name}"):
            st.json(
                {
                    "file_path": row.file_path,
                    "doc_type": row.doc_type,
                    "last_error": row.last_error,
                    "retry_count": row.retry_count,
                    "failed_at": row.failed_at,
                },
                expanded=True,
            )

# ──────────────────────────────────────────────
#  4. Self-Correction Audit Trail
# ──────────────────────────────────────────────
st.markdown(
    '<div class="section-title">🔍 Self-Correction Audit Trail</div>',
    unsafe_allow_html=True,
)

doc_ids = []
if len(df_corr):
    doc_ids = df_corr["doc_id"].unique().tolist()
elif len(df):
    doc_ids = df["doc_id"].unique().tolist()

if doc_ids:
    selected_doc_id = st.selectbox(
        "Select a doc_id to inspect its correction history",
        options=doc_ids,
        format_func=lambda x: f"{x[:20]}...",
    )

    corr_filtered = df_corr[df_corr["doc_id"] == selected_doc_id]

    if corr_filtered.empty:
        st.info(
            "ℹ️ This document passed all validation on the first attempt — "
            "no self-correction cycle was triggered.\n\n"
            "💡 Documents that fail validation will have their correction "
            "history displayed here."
        )

        ext_row = df[df["doc_id"] == selected_doc_id]
        if len(ext_row):
            st.markdown("**Extraction record for this document:**")
            st.dataframe(ext_row, width="stretch", height=120)
    else:
        st.success(
            f"📜 Document `{selected_doc_id[:20]}...` went through "
            f"{len(corr_filtered)} correction cycle(s)"
        )
        st.dataframe(corr_filtered, width="stretch")

        fig2 = px.bar(
            corr_filtered,
            x="cycle",
            y="cycle",
            hover_data={
                "error_summary": True,
                "llm_raw_response": True,
                "created_at": True,
            },
            labels={"cycle": "Correction Cycle"},
            title="Correction Cycles Distribution",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            showlegend=False,
        )
        st.plotly_chart(fig2, width="stretch")
else:
    st.info("ℹ️ No doc_id data available yet")

# ──────────────────────────────────────────────
#  5. Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🔐 Financial AI Pipeline · Senior AI Data Engineer · Industrial-Grade Unstructured Financial Data Extraction Platform"
)

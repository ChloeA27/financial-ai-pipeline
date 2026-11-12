"""
AI Pipeline Monitoring Dashboard (Streamlit)

Reads data/pipeline.db (or $PIPELINE_DB) on page load and displays:
  - Outcomes per document: stored / human review / out of scope / DLQ
    (``documents.last_status``, set by ``outcome_of``)
  - Human review queue: open items a person must decide on
  - Dead Letter Cemetery: items an engineer must act on
  - Self-Correction Audit Trail (per-doc_id drill-down)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.viz.pipeline_db import read_rows

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
# PIPELINE_DB points the page at another database, e.g. an E4 run's run.db.
DB_PATH = Path(
    os.environ.get(
        "PIPELINE_DB", Path(__file__).resolve().parent.parent / "data" / "pipeline.db"
    )
)


# ──────────────────────────────────────────────
#  DB helpers
# ──────────────────────────────────────────────
def query(sql: str) -> pd.DataFrame:
    """Run one read-only query (fresh connection per call, see src/viz/pipeline_db.py)."""
    if not DB_PATH.exists():
        st.error(f"❌ Database not found at {DB_PATH}")
        st.stop()
    columns, rows = read_rows(DB_PATH, sql)
    return pd.DataFrame(rows, columns=columns)


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
.review .value { color: #fab219; }
.neutral .value { color: #898781; }
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
df_docs = query("SELECT * FROM documents")
_tables = set(
    query("SELECT name FROM sqlite_master WHERE type = 'table'")["name"].tolist()
)
# The dashboard only reads: a database created before 2026-11-12 has no
# review_queue until the pipeline runs once (init_db adds it).
has_review_table = "review_queue" in _tables
df_review = (
    query("SELECT * FROM review_queue ORDER BY updated_at DESC")
    if has_review_table
    else pd.DataFrame()
)
df_open = df_review[df_review["status"] == "open"] if len(df_review) else df_review

# One outcome per document: its last run (documents.last_status)
status_counts = df_docs["last_status"].value_counts().to_dict() if len(df_docs) else {}
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

_cards = [
    ("total", len(df_docs), "📂 Documents"),
    ("pass", status_counts.get("stored", 0), "✅ Stored"),
    ("review", status_counts.get("human_review", 0), "🙋 Human review"),
    ("neutral", status_counts.get("out_of_scope", 0), "➖ Out of scope"),
    ("fail", status_counts.get("dlq", 0), "☠️ DLQ (engineering)"),
]
for col, (css, value, label) in zip(st.columns(len(_cards)), _cards):
    with col:
        st.markdown(
            f'<div class="card {css}"><p class="value">{value}</p>'
            f'<p class="label">{label}</p></div>',
            unsafe_allow_html=True,
        )
st.caption(
    f"Outcome of each document's last run. Avg retry count of stored assets: {avg_retry}. "
    "Documents last processed before 2026-11-12 may show 'dlq' for review cases."
)

# ──────────────────────────────────────────────
#  2. Version History (Plotly)
# ──────────────────────────────────────────────
st.markdown(
    '<div class="section-title">📈 Asset Version History</div>', unsafe_allow_html=True
)

if len(df):
    # One bar per document: how many gold versions it has (SCD Type 2), split
    # into the active version and archived ones. is_current is 0/1 in SQLite;
    # map it to labels so Plotly colours it as a category, not a number.
    versions = (
        df.assign(
            state=df["is_current"].map({1: "Active", 0: "Archived"}),
            document=df["file_path"].map(lambda p: Path(p).name),
        )
        .groupby(["document", "doc_id", "state"])
        .size()
        .reset_index(name="versions")
    )
    fig = px.bar(
        versions,
        x="versions",
        y="document",
        color="state",
        orientation="h",
        hover_data={"doc_id": True},
        labels={"versions": "Versions", "document": "", "state": ""},
        # Status colours: active = good; archived = neutral (normal, not an error)
        color_discrete_map={"Active": "#0ca30c", "Archived": "#898781"},
        category_orders={"state": ["Active", "Archived"]},
        title="Gold versions per document",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
        legend_title_text="",
        height=120 + 40 * versions["document"].nunique(),
    )
    fig.update_xaxes(dtick=1, gridcolor="rgba(137,135,129,0.25)")
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "A rerun on changed content archives the old version and adds a new active one "
        "(SCD Type 2). Rerunning unchanged content adds no version."
    )

    with st.expander("📋 View extractions raw data"):
        st.dataframe(df, width="stretch", height=300)
else:
    st.info("ℹ️ No data in extractions table yet")

# ──────────────────────────────────────────────
#  3. Human review queue (a person decides)
# ──────────────────────────────────────────────
st.markdown(
    '<div class="section-title">🙋 Human Review Queue — a person must decide</div>',
    unsafe_allow_html=True,
)

if not has_review_table:
    st.info(
        "This database has no review_queue table yet: it was created before "
        "2026-11-12. Run the pipeline once on it (for example "
        "`python -m src.main --dir <folder>`) and init_db adds the table."
    )
elif df_open.empty:
    st.success("No open review items.")
else:
    st.warning(f"{len(df_open)} document(s) wait for a person:")
    st.dataframe(
        df_open[
            [
                "file_path",
                "doc_type",
                "reason_stage",
                "reason_code",
                "reason_category",
                "updated_at",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    for _, row in df_open.iterrows():
        with st.expander(f"🙋 {Path(row.file_path).name} — {row.reason_code}"):
            st.write(row.reason_message)
            for col in ("extracted_data", "classification", "failure_details"):
                value = row.get(col)
                if isinstance(value, str) and value:
                    st.caption(col)
                    st.code(value, language="json")

# ──────────────────────────────────────────────
#  4. Dead Letter Cemetery (an engineer acts)
# ──────────────────────────────────────────────
st.markdown(
    '<div class="section-title">☠️ Dead Letter Cemetery — an engineer must act</div>',
    unsafe_allow_html=True,
)

# A DLQ row stays as history after a later run settles the document; show only
# documents whose LAST run ended in the DLQ.
_dlq_now = set(df_docs.loc[df_docs["last_status"] == "dlq", "source_hash"])
df_dlq_now = df_dlq[df_dlq["source_hash"].isin(_dlq_now)] if len(df_dlq) else df_dlq
if len(df_dlq) > len(df_dlq_now):
    st.caption(
        f"{len(df_dlq) - len(df_dlq_now)} older DLQ row(s) hidden: a later run of "
        "the same content ended elsewhere (stored, review or out of scope)."
    )
if df_dlq_now.empty:
    st.success("No document is in the DLQ.")
else:
    st.error(f"⚠️ {len(df_dlq_now)} document(s) need an engineer:")
    for _, row in df_dlq_now.iterrows():
        with st.expander(f"💥 {Path(row.file_path).name}"):
            st.json(
                {
                    "file_path": row.file_path,
                    "doc_type": row.doc_type,
                    "failed_stage": (
                        None
                        if pd.isna(row.get("failed_stage"))
                        else row.get("failed_stage")
                    ),
                    "error_code": (
                        None
                        if pd.isna(row.get("error_code"))
                        else row.get("error_code")
                    ),
                    "error_category": (
                        None
                        if pd.isna(row.get("error_category"))
                        else row.get("error_category")
                    ),
                    "last_error": row.last_error,
                    "retry_count": row.retry_count,
                    "failed_at": row.failed_at,
                },
                expanded=True,
            )

# ──────────────────────────────────────────────
#  5. Self-Correction Audit Trail
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

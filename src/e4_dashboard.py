"""
E4 end-to-end dashboard: where each real filing ended up.

    make dash            # or: python -m streamlit run src/app.py

Reads one run directory under ``results/e4/`` (written by ``evals/run_e4.py``).
Data shaping lives in ``src/viz/e4_data.py`` (tested without Streamlit).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.viz import e4_data as d  # noqa: E402

RESULTS = Path(os.environ.get("E4_RESULTS_DIR", ROOT / "results" / "e4"))

# Outcomes are states: status palette, always shown with a text label (legend,
# hover, table). Same steps as the E1 dashboard's outcome colours.
OUTCOME_COLORS = {
    "Stored": "#0ca30c",  # good
    "Human review": "#fab219",  # warning: a person must act
    "Out of scope": "#898781",  # neutral: expected, not a failure
    "DLQ (engineering)": "#d03b3b",  # critical
}
SCOPE_LABELS = {
    "all": "All labels",
    "verified": "Human-verified labels only",
    "silver": "Silver (draft) labels only",
}


def _style(fig):
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="",
        bargap=0.3,
        hoverlabel=dict(namelength=-1),
    )
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def _pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


runs = d.list_runs(RESULTS)
if not runs:
    st.title("E4 · end-to-end funnel")
    st.warning(f"No E4 runs under {RESULTS}. Run `python -m evals.run_e4` first.")
    st.stop()

with st.sidebar:
    st.header("Run")
    run_dir = st.selectbox("E4 run", runs, format_func=lambda p: p.name)
    scope_key = st.radio(
        "Labels", list(SCOPE_LABELS), format_func=SCOPE_LABELS.get, index=0
    )
    m = d.load_manifest(run_dir)
    git = m.get("git", {})
    st.caption(
        f"**Model** {m.get('model')} · **classifier input** "
        f"{m.get('classifier_input_strategy')} · **thinking** {m.get('classify_thinking')}  \n"
        f"**Cases** {m.get('cases')} · split {m.get('split')}  \n"
        f"**Commit** `{str(git.get('commit', ''))[:8]}`"
        f"{' (dirty)' if git.get('dirty') else ''}  \n**Created** {m.get('created_at_utc')} UTC"
    )

summary = d.load_summary(run_dir)
scope = summary["scopes"][scope_key]
k = d.kpis(scope)

st.title("E4 · end-to-end funnel")
st.caption(
    "The whole pipeline on real EDGAR dev filings. Classification is scored against the "
    "labels in scope. Field values are not scored here (that is E2). Silver labels are "
    "drafts: treat silver numbers as indicative."
)

c = st.columns(6)
c[0].metric("Documents", k["documents"])
c[1].metric("Classification correct", _pct(k["classification_accuracy"]))
c[2].metric("Stored", k["stored"])
c[3].metric("Human review", k["human_review"])
c[4].metric("Out of scope", k["out_of_scope"])
c[5].metric("DLQ", k["dlq"])
st.caption(
    f"Supported filings (gold M&A or Dividend) stored without a person: "
    f"**{_pct(k['supported_stored_rate'])}** · "
    f"{summary['llm_tokens']['prompt']:,} prompt + "
    f"{summary['llm_tokens']['completion']:,} completion tokens · "
    f"{summary['wall_seconds']} s"
)

st.subheader("Where each kind of filing ended up")
rows = pd.DataFrame(d.funnel_rows(scope))
if rows.empty:
    st.info("No documents in this scope.")
else:
    fig = px.bar(
        rows,
        x="count",
        y="gold",
        color="outcome",
        orientation="h",
        color_discrete_map=OUTCOME_COLORS,
        category_orders={
            "outcome": list(OUTCOME_COLORS),
            "gold": list(dict.fromkeys(rows["gold"])),
        },
        custom_data=["share", "total"],
        labels={"count": "Documents", "gold": "Gold label", "outcome": "Outcome"},
    )
    fig.update_traces(
        hovertemplate="%{y} → %{fullData.name}: %{x} of %{customdata[1]} "
        "(%{customdata[0]:.0%})<extra></extra>"
    )
    st.plotly_chart(_style(fig), width="stretch")
    with st.expander("Table"):
        st.dataframe(rows, hide_index=True, width="stretch")

st.subheader("Why documents went to a person or the DLQ")
reasons = pd.DataFrame(d.reason_rows(scope))
if reasons.empty:
    st.success("No document went to human review or the DLQ.")
else:
    st.dataframe(reasons, hide_index=True, width="stretch")

st.subheader("Documents")
cases, full = d.load_cases(run_dir)
if not full:
    st.caption(
        "Showing the committed per_case.csv: unverified cases carry no labels or answers."
    )
df = pd.DataFrame(cases)
if not df.empty:
    if scope_key != "all" and "verified" in df:
        df = df[df["verified"] == (scope_key == "verified")]
    if "outcome" in df:
        pick = st.multiselect(
            "Outcome", sorted(df["outcome"].dropna().unique()), placeholder="All"
        )
        if pick:
            df = df[df["outcome"].isin(pick)]
    st.dataframe(df, hide_index=True, width="stretch")

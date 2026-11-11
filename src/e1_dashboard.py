"""
E1 classification-evaluation dashboard.

    streamlit run src/e1_dashboard.py

Three levels, all read from one run directory under ``results/e1/``:

  1. Compare variants (meso)  — the paired comparison of input strategies
  2. Outcome funnel (macro)   — where classification runs end, why attempts
                                were rejected, how many tokens calls use
  3. Document trace (micro)   — one document's attempts, the raw model replies,
                                and the window it saw with the quotes marked

Data shaping lives in ``src/viz/e1_data.py`` (tested without Streamlit).
Cases that are not human-verified are hidden in the trace view unless you
opt in — seeing model answers would anchor blind labelling.
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

from src.viz import e1_data as d  # noqa: E402

RESULTS = Path(os.environ.get("E1_RESULTS_DIR", ROOT / "results" / "e1"))

# Categorical slots 1–3 of the reference palette (validated light + dark).
VARIANT_COLORS = {
    "light": {"head": "#2a78d6", "structured": "#eb6834", "keyword": "#1baf7a"},
    "dark": {"head": "#3987e5", "structured": "#d95926", "keyword": "#199e70"},
}
# Outcomes are states → status palette, always shown with a text label.
OUTCOME_COLORS = {
    "accepted · 1st attempt": "#0ca30c",  # good
    "accepted · after retry": "#fab219",  # warning
    "ended · none": "#898781",  # neutral: expected, not a failure
    "ended · unclear": "#ec835a",  # serious
    "failed": "#d03b3b",  # critical
}

st.set_page_config(page_title="E1 classification eval", page_icon="🧪", layout="wide")


def _mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit or unknown theme
        return "light"


def _variant_colors() -> dict[str, str]:
    colors = VARIANT_COLORS[_mode()]
    return {d.VARIANT_LABELS[k]: v for k, v in colors.items()}


def _style(fig):
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="",
        bargap=0.25,
        bargroupgap=0.08,
        hoverlabel=dict(namelength=-1),
    )
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=False)
    return fig


# ── Run selection ────────────────────────────────────────────────────

runs = d.list_runs(RESULTS)
if not runs:
    st.warning(f"No E1 runs found under {RESULTS}. Run evals/run_e1.py first.")
    st.stop()

with st.sidebar:
    st.header("Run")
    run_dir = st.selectbox("E1 run", runs, format_func=lambda p: p.name)
    run = d.load_run(run_dir)
    m = run["manifest"]
    st.caption(
        f"**Model** {m.get('model')} · **temperature** {m.get('temperature')}  \n"
        f"**Budget** {m.get('budget_tokens')} tokens · **reps** {m.get('reps')}  \n"
        f"**Cases** {len(m['cases'])} ({sum(c['verified'] for c in m['cases'])} verified) "
        f"· split {m.get('split')}  \n"
        f"**Commit** `{m['git']['commit'][:8]}`{' (dirty)' if m['git']['dirty'] else ''}  \n"
        f"**Created** {m.get('created_at', '')[:16]} UTC"
    )
    st.caption(f"Projection: {m.get('projection_rule')}")

summary = run["summary"]
variants = d.variants_in(summary)
n_verified = sum(c["verified"] for c in m["cases"])

st.title("E1 · classification stage")
st.caption(
    f"Offline paired comparison of classifier input strategies on {len(m['cases'])} frozen "
    f"dev cases. Accuracy counts only the {n_verified} human-verified cases; the rest "
    "contribute label-free metrics."
)

tab_compare, tab_funnel, tab_trace = st.tabs(
    ["① Compare variants", "② Outcome funnel", "③ Document trace"]
)

# ── ① Compare variants ───────────────────────────────────────────────

with tab_compare:
    cols = st.columns(len(variants))
    for col, v in zip(cols, variants):
        s = summary["variants"][v]
        col.metric(
            d.VARIANT_LABELS[v],
            d.fmt(s.get("accuracy_verified"), "rate"),
            help="Accuracy on verified cases × repetitions",
        )
        col.caption(
            f"{d.fmt(s.get('classify_failed_rate'), 'rate')} failed · "
            f"{d.fmt(s.get('tokens_per_doc_mean'), 'tokens')} tokens/doc"
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Quality and reliability")
        df = pd.DataFrame(
            d.rate_series(
                summary,
                [
                    "accuracy_verified",
                    "input_coverage_verified",
                    "classify_retry_rate",
                    "classify_failed_rate",
                    "rep_agreement",
                ],
            )
        )
        df["variant"] = df["variant"].map(d.VARIANT_LABELS)
        fig = px.bar(
            df,
            x="metric",
            y="value",
            color="variant",
            barmode="group",
            text=df["value"].map(lambda x: f"{x:.0%}"),
            color_discrete_map=_variant_colors(),
            labels={"value": "", "metric": ""},
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
        st.plotly_chart(_style(fig), width="stretch")
    with right:
        st.subheader("Cost per document")
        rows = []
        for v in variants:
            s = summary["variants"][v]
            rows.append(
                {
                    "variant": d.VARIANT_LABELS[v],
                    "tokens": s.get("tokens_per_doc_mean"),
                    "latency_s": (s.get("latency_ms_per_doc_mean") or 0) / 1000,
                }
            )
        cost = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        for col, field, title, fmt in (
            (c1, "tokens", "Tokens per document", "{:,.0f}"),
            (c2, "latency_s", "Latency per document (s)", "{:.1f}"),
        ):
            fig = px.bar(
                cost,
                x="variant",
                y=field,
                color="variant",
                text=cost[field].map(fmt.format),
                color_discrete_map=_variant_colors(),
                labels={field: "", "variant": ""},
                title=title,
            )
            fig.update_layout(showlegend=False)
            col.plotly_chart(_style(fig), width="stretch")

    st.subheader("All metrics")
    st.dataframe(
        pd.DataFrame(d.metric_table(summary)), hide_index=True, width="stretch"
    )

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Accuracy by group")
        st.dataframe(
            pd.DataFrame(d.group_table(summary)), hide_index=True, width="stretch"
        )
        st.caption(
            "Groups other than `typical` are blind-labelled and not verified yet, so they "
            "cannot be scored. They are where input strategies are expected to differ."
        )
    with g2:
        st.subheader("Paired comparison (repetition 1)")
        st.dataframe(
            pd.DataFrame(d.paired_table(summary)), hide_index=True, width="stretch"
        )
        st.caption(
            "Only discordant pairs carry evidence. With ~17 verified cases only large "
            "differences can reach significance."
        )

# ── ② Outcome funnel ─────────────────────────────────────────────────

with tab_funnel:
    if run["db"] is None:
        st.info("run.db is not available locally for this run (it is not committed).")
    else:
        rows = d.step_rows(run["db"])
        index = d.run_index(run["per_case"])

        st.subheader("Where classification runs end")
        st.caption(
            "Every document × repetition, all 35 cases (aggregate only). "
            "`ended · none` is an expected outcome (mgmt_change and other filings), not a failure."
        )
        outcomes = d.outcomes_by_variant(rows, index)
        long = [
            {
                "variant": d.VARIANT_LABELS[v],
                "outcome": o,
                "runs": outcomes[v].get(o, 0),
            }
            for v in variants
            for o in d.OUTCOME_ORDER
            if outcomes[v].get(o, 0)
        ]
        fig = px.bar(
            pd.DataFrame(long),
            y="variant",
            x="runs",
            color="outcome",
            orientation="h",
            text="runs",
            category_orders={"outcome": list(d.OUTCOME_ORDER)},
            color_discrete_map=OUTCOME_COLORS,
            labels={"runs": "runs", "variant": ""},
        )
        fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.25))
        fig.update_traces(marker_line_color="rgba(252,252,251,1)", marker_line_width=2)
        st.plotly_chart(_style(fig), width="stretch")

        a, b = st.columns(2)
        with a:
            st.subheader("Why attempts were rejected")
            causes = d.retry_causes(rows, index)
            crow = [
                {"variant": d.VARIANT_LABELS[v], "code": code, "attempts": n}
                for v in variants
                for code, n in causes.get(v, {}).items()
            ]
            if crow:
                cdf = pd.DataFrame(crow)
                fig = px.bar(
                    cdf,
                    x="variant",
                    y="attempts",
                    color="variant",
                    text="attempts",
                    hover_data=["code"],
                    color_discrete_map=_variant_colors(),
                    labels={"variant": "", "attempts": "rejected attempts"},
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(_style(fig), width="stretch")
                st.dataframe(cdf, hide_index=True, width="stretch")
            else:
                st.success("No rejected attempts.")
        with b:
            st.subheader("Output tokens per classifier call")
            calls = pd.DataFrame(d.llm_calls(rows, index))
            calls["variant"] = calls["variant"].map(d.VARIANT_LABELS)
            calls["reply"] = calls["parsed"].map({True: "parsed", False: "unparsable"})
            fig = px.strip(
                calls,
                x="variant",
                y="completion_tokens",
                color="variant",
                hover_data=["reply", "prompt_tokens"],
                color_discrete_map=_variant_colors(),
                labels={"completion_tokens": "completion tokens", "variant": ""},
            )
            fig.add_hline(
                y=1024,
                line_dash="dot",
                line_color="#898781",
                annotation_text="max_tokens = 1024",
                annotation_position="top left",
            )
            fig.update_traces(marker_size=8)
            fig.update_layout(showlegend=False)
            st.plotly_chart(_style(fig), width="stretch")
            capped = int((calls["completion_tokens"] >= 1024).sum())
            st.caption(
                f"{capped} of {len(calls)} calls hit the 1,024-token cap — every one of them "
                "returned an empty or truncated reply. The visible reply is only ~300 "
                "characters, so most completion tokens are hidden reasoning."
            )

# ── ③ Document trace ─────────────────────────────────────────────────

with tab_trace:
    if run["db"] is None:
        st.info("run.db is not available locally for this run (it is not committed).")
        st.stop()
    rows = d.step_rows(run["db"])
    include = st.toggle(
        "Include cases that are not human-verified",
        value=False,
        help="Off by default: seeing the model's answers would anchor blind labelling.",
    )
    if include:
        st.warning("Showing model answers on cases you have not labelled yet.")
    cases = d.selectable_cases(run["per_case"], include)
    case_ids = sorted({c["case_id"] for c in cases})
    c1, c2, c3 = st.columns([2, 2, 1])
    case_id = c1.selectbox("Case", case_ids)
    variant = c2.selectbox(
        "Variant", variants, format_func=lambda v: d.VARIANT_LABELS.get(v, v)
    )
    reps = sorted(
        {
            int(c["rep"])
            for c in cases
            if c["case_id"] == case_id and c["variant"] == variant
        }
    )
    rep = c3.selectbox("Repetition", reps)
    case = next(
        c
        for c in cases
        if c["case_id"] == case_id and c["variant"] == variant and int(c["rep"]) == rep
    )
    steps = d.trace(rows, case["run_id"])

    verified = case.get("verified") == "True"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Gold (projected)", case.get("gold_projected") or "—" if verified else "hidden"
    )
    k2.metric("Answer", case.get("predicted") or "—")
    k3.metric(
        "Correct",
        {"True": "✅ yes", "False": "❌ no"}.get(case.get("correct", ""), "—"),
    )
    k4.metric("Classifier attempts", case.get("classify_attempts"))

    st.subheader("Attempts")
    badge = {
        "passed": "✅ passed",
        "rejected": "🔁 rejected → retry",
        "fatal": "⛔ ended here",
    }
    for s in steps:
        with st.container(border=True):
            top = st.columns([2, 2, 3, 2, 2])
            top[0].markdown(f"**{s['stage']}** · attempt {s['attempt']}")
            top[1].markdown(badge.get(s["outcome"], s["outcome"]))
            top[2].markdown(f"`{s['error_code']}`" if s.get("error_code") else "")
            if s.get("total_tokens") is not None:
                top[3].markdown(
                    f"{s['prompt_tokens']:,} in · {s['completion_tokens']:,} out"
                )
            if s.get("latency_ms") is not None:
                top[4].markdown(f"{s['latency_ms'] / 1000:.1f} s")
            if s.get("raw_response") is not None:
                with st.expander("Raw model reply"):
                    st.code(s["raw_response"] or "(empty reply)", language="json")
            if s.get("output"):
                with st.expander("Recorded output"):
                    st.json(s["output"], expanded=False)

    st.subheader("What the classifier saw")
    win = d.window_for(run["dir"], steps, int(m.get("budget_tokens") or 3500))
    if win is None:
        st.info("The stored source text is not available for this run.")
    else:
        quotes = []
        model_quotes = [
            (s.get("output") or {}).get("evidence_quote")
            for s in steps
            if s["stage"] == "classify"
        ]
        model_quote = next((q for q in reversed(model_quotes) if q), None)
        if model_quote:
            quotes.append((model_quote, "model"))
        body, found = d.highlight(win["text"], quotes)
        st_ = win["stats"]
        st.caption(
            f"Strategy **{st_.get('strategy')}** · sent {st_.get('chars_sent', 0):,} of "
            f"{st_.get('chars_total', 0):,} characters · truncated {st_.get('truncated')} · "
            f"sections: {', '.join(st_.get('sections') or [])}"
        )
        if model_quote:
            st.markdown(
                "<span class='legend model'>model's evidence quote</span>"
                + ("" if found[0] else " — *not found in the window*"),
                unsafe_allow_html=True,
            )
        st.markdown(
            """
<style>
.window {max-height: 520px; overflow-y: auto; white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem;
  line-height: 1.45; padding: 12px 14px; border-radius: 6px;
  border: 1px solid rgba(137,135,129,0.35);}
mark.model, .legend.model {background: rgba(250,178,25,0.45); color: inherit;
  padding: 0 2px; border-radius: 3px;}
</style>
""",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='window'>{body}</div>", unsafe_allow_html=True)

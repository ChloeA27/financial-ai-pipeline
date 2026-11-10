# Project summary — state as of 2026-11-10

Branch `feat/classification-validation-and-dlq-stages` (not yet merged to `main`).
Every number below comes from a rerunnable script; raw outputs live under `results/e1/`.

## What changed (2026-11-10)

| Commit | Change |
|---|---|
| `9858f4e` | Classify loop: enum labels (`M&A`, `Dividend`, `none`, `unclear`), verbatim `evidence_quote`, a code-only Classification Validator, own retry budget. Stage-tagged DLQ (`failed_stage`, `error_code`, `error_category`, …). Medallion storage: bronze raw store + `documents`, silver `step_runs` (one row per node execution, raw LLM reply, tokens, latency, prompt hash), gold `extractions` with `run_id`. 500-char snippet removed. `RAG_ENABLED=false` by default. |
| `0e58edc` | Byte-exact ingestion: original bytes stored and hashed before decoding; UTF-8 → Windows-1252 fallback recorded; undecodable/binary files go to the DLQ (`read` / `DECODE_FAILED`) instead of being skipped. |
| `9490039` | Classifier input strategies `head` (V0, default) / `structured` (V1) / `keyword` (V2); evidence quote checked inside the window actually sent (≥ 25 chars, ≥ 4 words) with diagnostics; E1 runner `scripts/eval_classification.py`. |
| `d7a23a7` | First E1 run (below); model answers on unverified cases sealed in `_sealed/`. |
| `11ce062` | Fix: concurrent writers of identical content raced on one temporary file in the raw store. |
| `302625b` | E1 dashboard (`streamlit run src/e1_dashboard.py`): variant comparison, outcome funnel, per-document trace. |

Unit tests: 216 → 412 (no LLM calls). `black --check` clean. CI on Python 3.11 not yet run for this branch.

## E1 — classification stage, first run

35 original dev cases of the E1 SEC EDGAR set (17 human-verified, all in the `typical` group),
`deepseek-flash`, temperature 0, 2 repetitions per variant, budget 3,500 tokens.
233 classifier calls, 479,182 prompt + 81,271 completion tokens. Gold labels are projected onto
`{mna, dividend}` (`mgmt_change`-only filings count as `none`).
Source: `results/e1/20261110-214827_dev35_flash/`.

| Metric | V0 head | V1 structured | V2 keyword |
|---|---|---|---|
| Accuracy, verified cases × reps | 33/34 | 34/34 | 34/34 |
| Input coverage, verified positives (n=8) | 0.875 | 1.0 | 1.0 |
| Documents failing classification (all 35) | 2.9% | 0% | 1.4% |
| Documents needing a classify retry | 10% | 4.3% | 10% |
| Agreement between repetitions | 94.3% | 100% | 97.1% |
| Tokens per document | 2,319 | 2,934 | 2,754 |
| Latency per document | 3.0 s | 2.3 s | 3.1 s |

How to read it:

- No difference between variants is demonstrated: McNemar on repetition 1 has no discordant pairs.
  The verified cases are the easy `typical` group (ceiling effect); the groups where input selection
  should matter (`long`, `format`, `confuser`, `multi`) are blind-labelled and not verified yet.
- V0's only verified miss (E1-023) is an output truncation, not a window problem.
- Coverage-only comparison (no LLM; `results/e1/20261110-214025_coverage_orig35/`), on draft
  (unverified) positives n=16: V0 0.625, V1 0.9375, V2 0.875, head truncated at the same budget
  0.6875 — the gain comes from selection, not from a larger budget. Indicative only.

## Open issues

1. **Output truncation.** 26 of 233 calls used exactly `max_tokens=1024` completion tokens and returned
   an empty or cut-off reply; successful calls also spend up to ~1,000 completion tokens on ~300 visible
   characters (hidden reasoning in `deepseek-flash`). Fix options: raise `max_tokens`, or disable
   thinking for the classifier if the API supports it; then rerun E1.
2. **Model name.** `deepseek-chat` is deprecated by the provider (announced discontinuation
   2026-07-24) but is still the default in `src/config.py` and in CI's golden evaluation. E1 used
   `deepseek-flash` via an environment override.
3. **Labels.** Blind groups of the E1 dev split still need human labels before accuracy can be
   reported for them; the expanded dataset (dev 100) is not labelled.
4. **Not built:** LLM-as-judge, `mgmt_change` extraction, multi-label routing, extraction-stage
   evaluation (E2), end-to-end funnel (E4), throughput measurement.
5. **Not verified:** dashboard layout by eye (rendered headlessly with Streamlit `AppTest` only);
   Python 3.11.

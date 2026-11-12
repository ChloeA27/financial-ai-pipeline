# Project summary — state as of 2026-11-11

Branch `feat/classification-validation-and-dlq-stages` (not yet merged to `main`).
Every number below comes from a rerunnable script. Raw outputs live under `results/e1/`
(index: `results/e1/README.md`).

## What changed

| Commit | Change |
|---|---|
| `9858f4e` | Classify loop: enum labels (`M&A`, `Dividend`, `none`, `unclear`), verbatim `evidence_quote`, a code-only Classification Validator, and its own retry budget. Stage-tagged DLQ (`failed_stage`, `error_code`, `error_category`, …). Medallion storage: bronze raw store + `documents`; silver `step_runs` (one row per node execution with raw LLM reply, tokens, latency, prompt hash); gold `extractions` with `run_id`. 500-char snippet removed. `RAG_ENABLED=false` by default. |
| `0e58edc` | Byte-exact ingestion: original bytes are stored and hashed before decoding. The UTF-8 → Windows-1252 fallback is recorded. Undecodable or binary files go to the DLQ (`read` / `DECODE_FAILED`) instead of being skipped. |
| `9490039` | Classifier input strategies `head` (V0, default) / `structured` (V1) / `keyword` (V2). The evidence quote is checked inside the window actually sent (≥ 25 chars, ≥ 4 words), with diagnostics. E1 runner `scripts/eval_classification.py`. |
| `d7a23a7` | First E1 run. Model answers on unverified cases are sealed in `_sealed/`. |
| `11ce062` | Fix: concurrent writers of identical content raced on one temporary file in the raw store. |
| `302625b` | E1 dashboard (`streamlit run src/e1_dashboard.py`): variant comparison, outcome funnel, per-document trace. |
| `ea0d32c`, `fd9869d` | Repo tidy: tests mirror `src/`; docs match the current layout; index of E1 runs. |
| `f5fb99b` | E1 runner reads the dataset path from `E1_DATASET` (the dataset stays outside the repo and is only read). |
| `b0db1ee` | **Classifier thinking mode off by default** (`CLASSIFY_THINKING=disabled`), found by the first E1 run (below). |
| `8d35d12` | E1 run 2 (thinking off) and the iteration notes. |
| `c554157` … `ef8aad0`, then the `evals/` move | Layered structure, moves only (no behaviour change): prompts in `src/prompts/templates/` (byte-identical, same prompt fingerprint); contracts in `src/schemas/`; routing (`src/workflows/edges.py`) split from wiring (`graph.py`); `validator` renamed `extraction_validator`; evaluation in a top-level `evals/` (datasets, evaluators, `run_golden.py`, `run_e1.py`). Layering rules enforced by `tests/test_workflows/test_layering.py`. |
| `be01628`, `53fa7e5` | E1 scorer self-checks (gold as output → 100 %, constant `none` → only where `none` is acceptable, API outage → `error`, not wrong). They found that per-label recall counted an accepted `unclear` as a miss; it is now a hand-off counted in `unclear_accepted` (runs 1–2 unchanged). Golden set marked synthetic. |

Tests: 216 → 435 collected (2026-11-12). 431 run offline with a scripted fake LLM — this is
what CI runs (`pytest -m "not integration"`). The other 4 are marked `integration` and call the
real DeepSeek API when a key is set, so a bare `pytest` is not free. `black --check` clean. CI on Python 3.11 not yet run for this branch.

## E1 — classification stage

35 original dev cases of the E1 SEC EDGAR set (17 human-verified, all in the `typical` group),
`deepseek-flash`, 2 repetitions per variant, budget 3,500 tokens, `max_tokens` 1,024. Gold labels
are projected onto `{mna, dividend}` (`mgmt_change`-only filings count as `none`).

### Iteration found by the experiment

1. **Run 1** (`results/e1/20261110-214827_dev35_flash/`, thinking on, the model default):
   26 of 233 classifier calls stopped at exactly 1,024 completion tokens with an empty or cut-off
   reply. `deepseek-flash` thinks before answering, and the hidden reasoning counts toward
   `max_tokens`. DeepSeek's docs also say temperature has no effect in thinking mode, so this run's
   temperature 0 was not applied.
2. **Fix:** turn thinking off for the classifier. Classification needs no long reasoning, and
   with thinking off temperature applies again. Raising `max_tokens` would have hidden the
   symptom but kept paying for the reasoning.
3. **Run 2** (`results/e1/20261111-211954_dev35_flash_nothink/`). Only thinking changed: same
   cases, inputs, prompt, variants and limits.

| Metric (run 1 → run 2) | V0 head | V1 structured | V2 keyword |
|---|---|---|---|
| Accuracy, verified cases × reps | 33/34 → **34/34** | 34/34 → 34/34 | 34/34 → 34/34 |
| Input coverage, verified positives (n=8) | 0.875 | 1.0 | 1.0 |
| Documents failing classification (all 35) | 2.9% → **0%** | 0% → 0% | 1.4% → **0%** |
| Documents needing a classify retry | 10% → **1.4%** | 4.3% → 4.3% | 10% → **2.9%** |
| Agreement between repetitions | 94.3% → **100%** | 100% → 100% | 97.1% → 97.1% |
| Tokens per document | 2,319 → 1,682 | 2,934 → 2,723 | 2,754 → 2,084 |
| Latency per document | 3.0 s → 1.3 s | 2.3 s → 1.4 s | 3.1 s → 1.4 s |

Run 2 totals: 216 calls, 437,111 prompt + 17,074 completion tokens (run 1: 81,271).
The largest reply was 166 tokens, and no call hit the limit.

How to read it:

- Truncation is gone: parse failures dropped from 26 to 2. The remaining 2 quoted more than
  the schema's 400 characters, and both passed on retry. V0's only verified miss in run 1
  (E1-023) was a truncation and is correct in run 2.
- New and smaller: 4 `EVIDENCE_QUOTE_NOT_FOUND` rejections, all on E1-020, all passing on
  retry. The model dropped one comma from the source sentence. The strict verbatim check
  caught it. `longest_match_ratio` was 0.618, because the longest contiguous overlap breaks at
  the comma, so a low ratio alone does not prove an invented quote.
- No difference between variants is demonstrated: McNemar on repetition 1 has no discordant
  pairs in either run. The verified cases are the easy `typical` group (ceiling effect). The
  groups where input selection should matter (`long`, `format`, `confuser`, `multi`) are
  blind-labelled and not verified yet.
- Coverage-only comparison (no LLM; `results/e1/20261110-214025_coverage_orig35/`), on draft
  (unverified) positives n=16: V0 0.625, V1 0.9375, V2 0.875, and head truncated at the same
  budget 0.6875. The gain comes from selection, not from a larger budget. Indicative only.

## Open issues

1. **Model name.** `deepseek-chat` is deprecated by the provider (announced discontinuation
   2026-07-24) but is still the default in `src/config.py` and in CI's scenario gate. It
   accepts the thinking parameter (checked 2026-11-11). E1 used `deepseek-flash` via an
   environment override.
2. **Labels.** Blind groups of the E1 dev split still need human labels before accuracy can be
   reported for them. The expanded dataset (dev 100) is not labelled.
3. **Quote check strictness (decided 2026-11-11: stay strict).** A dropped comma fails the
   verbatim check (E1-020); in run 2 all 4 such rejections passed on retry. Punctuation is not
   relaxed.
4. **Golden gate design, redesigned 2026-11-12 on branch `feat/outcome-routing` (not merged).** The old gate pooled all checks, so the one expected_fail case could not fail it on its own (a wrongly stored document still scored 24/26 = 92 % > 85 %). It is now the *scenario gate* (`evals/run_scenarios.py`): 10 synthetic cases, one or two per path, each with an expected outcome (stored / out_of_scope / human_review / dlq). The outcome gate and the REQUIRED-field gate are separate. Not yet run with the real model.
5. **Not built:** LLM-as-judge, `mgmt_change` extraction, multi-label routing, extraction-stage
   evaluation (E2), end-to-end funnel (E4), throughput measurement.
6. **Not verified:** dashboard layout by eye (rendered headlessly with Streamlit `AppTest` only);
   Python 3.11.

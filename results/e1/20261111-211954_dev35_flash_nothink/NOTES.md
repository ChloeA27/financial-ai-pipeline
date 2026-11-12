# E1 run: 35 original dev cases, deepseek-flash, thinking disabled (2026-11-11, UTC 21:19)

Iteration on `../20261110-214827_dev35_flash/`. Only one setting changed: the classifier
now sends `{"thinking": {"type": "disabled"}}`. Everything else is the same: same 35 cases
and input files (identical `input_sha256`), the same 17 verified cases, prompt, variants,
2 repetitions, budget 3,500 tokens and `max_tokens` 1,024. The dataset file hash differs
because the labelling sheet was edited between runs. Verified labels are unchanged.

Why: in the first run, 26 of 233 classifier calls stopped at exactly `max_tokens`=1,024
with an empty or cut-off reply. `deepseek-flash` has thinking on by default, and its hidden
reasoning counts toward the output limit. DeepSeek's thinking-mode guide also says that
temperature "will not trigger an error but will also have no effect" in thinking mode, so
the first run's temperature 0 was not applied. Here it is (`temperature_applies: true`).

Code at run time: commit `f5fb99b` plus the uncommitted thinking-mode change (so the manifest
says `dirty: true`). That change was committed unchanged right after the run as `b0db1ee`.

## Result (first run → this run)

| | V0 head | V1 structured | V2 keyword |
|---|---|---|---|
| Accuracy, verified × reps | 33/34 → **34/34** | 34/34 → 34/34 | 34/34 → 34/34 |
| Docs needing a classify retry | 10% → **1.4%** | 4.3% → 4.3% | 10% → **2.9%** |
| Docs failing classification | 2.9% → **0%** | 0% → 0% | 1.4% → **0%** |
| Agreement between repetitions | 94.3% → **100%** | 100% → 100% | 97.1% → 97.1% |
| Tokens per document | 2,319 → 1,682 | 2,934 → 2,723 | 2,754 → 2,084 |
| Latency per document | 3.0 s → 1.3 s | 2.3 s → 1.4 s | 3.1 s → 1.4 s |

- Classifier calls: 233 → 216. Completion tokens: 81,271 → 17,074. Largest reply: 1,024 → 166
  tokens. Zero calls hit the limit.
  Cost ≈ $0.08 at the off-peak list price (≈ $0.15 at the peak price), before cache discounts.
- Parse failures: 26 → 2. The remaining 2 are not truncation: both `evidence_quote` values
  were longer than the schema's 400-character limit. Both passed on retry.
- New: 4 `EVIDENCE_QUOTE_NOT_FOUND` rejections, all on E1-020 (structured and keyword, both
  repetitions). Each passed on retry. The model dropped one comma from the source
  sentence (`(the “Company”), completed` → `(the “Company”) completed`).
  `longest_match_ratio` was 0.618 every time. The longest *contiguous* overlap breaks at
  the missing comma, so a low ratio does not by itself mean the quote was invented.
- Accepted quotes without label vocabulary: 10 → 0.
- Repetition disagreement: the 1 remaining pair (keyword) parsed both times and gave two
  different labels. Its case is not human-verified, so it is reported in aggregate only.
- McNemar on repetition 1: no discordant pairs. No difference between variants is shown.
  The verified cases are still only the easy `typical` group.
- The 18 unverified cases are aggregate-only. Their model answers are in `_sealed/`.

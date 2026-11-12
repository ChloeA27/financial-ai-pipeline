# E1 run — 35 original dev cases, deepseek-flash, 2026-11-10 (UTC 21:48, off-peak)

- 3 input strategies × 2 repetitions, temperature 0, budget 3,500 tokens; 233 classifier calls,
  479,182 prompt + 81,271 completion tokens (≈ $0.12 at off-peak list price, before cache discounts).
- Accuracy is scored only on the 17 human-verified cases (all in the `typical` group:
  8 positive after projection, 9 `mgmt_change` → `none`). head 33/34, structured 34/34,
  keyword 34/34; McNemar on repetition 1: no discordant pairs (no evidence of a difference).
- The only verified miss (E1-023, head, rep 2) is not a window problem: the classifier
  output was cut off at max_tokens=1024.
- Every CLASSIFICATION_PARSE_FAILED call (26 of 233) used exactly 1,024 completion tokens
  with an empty or truncated reply; successful calls use up to ~1,000 completion tokens for
  ~300 visible characters — deepseek-flash spends hidden reasoning tokens before answering.
- E1-024: gold evidence is outside the head window, yet head answered `mna` correctly in both
  repetitions with another valid quote → gold-evidence coverage is a conservative proxy.
- The 18 unverified cases are aggregate-only here; their model answers are in `_sealed/`.

## Addendum 2026-11-11

- `deepseek-flash` runs in thinking mode by default, and DeepSeek's thinking-mode guide says
  temperature "will not trigger an error but will also have no effect" there. The
  `temperature: 0` recorded in this run's manifest was therefore not applied.
- Rerun with thinking disabled (the only change): `../20261111-211954_dev35_flash_nothink/`.

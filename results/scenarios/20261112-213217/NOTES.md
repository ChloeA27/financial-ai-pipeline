# Scenario gate after the retry policy (2026-11-12)

Model `deepseek-chat`, commit `b75864b` (clean). The only change since `20261112-211858` that
affects the model's path: the retry policy (`missing_info` and `business_rule` are not retried).
Also changed: S1 now accepts both spellings of the acquirer, and H2/H3 now also check
`extract_attempts == 1`.

**Gate: PASS**, outcome gate 8/8. Pending S3 and S4 still fail, as expected.

| | Run 1 (`211858`) | Run 2 (`213217`) |
|---|---|---|
| Gate | FAIL | PASS |
| False accepts | 1 (H2) | 0 |
| H2 | stored a description as `target` after one feedback retry | human_review, `TARGET.NON_EMPTY`, 1 extraction |
| LLM calls | 26 | 15 |
| Prompt + completion tokens | 25,992 + 2,057 | 13,378 + 1,044 |

Fewer calls: S3 and S4 (missing optional-to-be fields) and H2 no longer spend 3 retries on a
`missing_info` failure. These are 10 synthetic cases, so the token drop shows the mechanism. It
is not a production estimate.

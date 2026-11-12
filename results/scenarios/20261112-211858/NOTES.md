# Scenario gate, first run with the real model (2026-11-12)

Model `deepseek-chat`, commit `44134e5` (clean). 26 LLM calls, 25,992 prompt + 2,057 completion
tokens. D1 needs no LLM call.

**Gate: FAIL**. Outcome gate 7/8. Pending S3 and S4 failed, as expected.

| Case | Result | Why |
|---|---|---|
| H2 | **False accept** (stored, should go to human review) | First extraction left `target` empty (honest). The validator rejected `TARGET.NON_EMPTY` and retried with that feedback. The second extraction wrote `"Privately held manufacturer of industrial pressure sensors (name not disclosed)"` into `target`, passed `non_empty`, and was stored. Confirmed by rerunning H2 alone (step trail: validate attempt 1 rejected, attempt 2 passed). **Evidence for the retry policy: no retry for `missing_info`, because feedback pushes the model to fill the field.** |
| S1 | Fail on `acquirer` | Test-data bug, not a model error. The source says both "Microsoft Corp." and "Microsoft Corporation"; the case accepted only the first. Fixed after this run with a `one_of` override. |
| S3, S4 | Pending, fail (human_review, `missing_info`) | Expected until `presence_levels` is implemented. The validator still requires `total_value_usd` and `ex_dividend_date`. |
| D1, H1, H3, O1, O2, S2 | Pass | Includes H1: the classifier said `unclear` for an acquisition plus a dividend. |

`per_case.json` was written before `stored_fields` existed. Later runs record what a falsely
accepted document stored.

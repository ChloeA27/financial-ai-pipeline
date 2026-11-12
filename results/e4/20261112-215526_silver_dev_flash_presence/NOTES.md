# E4 run 2: after presence levels in the validator (2026-11-12)

Same cases, model and settings as run 1 (`20261112-214732_silver_dev_flash`). The one change
that affects the pipeline: the validator reads the presence levels (commit `09e82d7`). An
empty non-REQUIRED field (deal value, ex-dividend date) no longer sends a filing to review.
Commit `340180d`. 270,296 prompt + 47,919 completion tokens, 88 s.

| | Run 1 | Run 2 |
|---|---|---|
| Supported filings (gold M&A / Dividend / both, n=42) stored | 2 (4.8 %) | **35 (83.3 %)** |
| Human review, all 100 | 40 | 4 |
| Of which validate `missing_info` on optional fields | 36 | 0 |
| Out of scope / DLQ | 56 / 0 | 56 / 0 |
| Classification, human-verified (n=17) | 17/17 | 17/17 |
| Classification vs silver labels (n=83) | 86.8 % | 86.8 % |

Remaining reviews: 3 × `RECORD_DATE.DATE_FORMAT` (REQUIRED record date empty), 1 × `unclear`.

## Read with care

- **Stored is not correct.** Field values are not scored (no field gold, see
  `evals/E2_DESIGN.md`). Storing more records also means more chances to store a wrong one.
  E2 has to measure that.
- **3 filings with silver label `none` were stored as M&A** (E1-093, E1-094, E1-100; all
  unverified). One of them is an 8-K/A that furnishes the financials of an acquisition
  completed earlier, which the labelling guide calls `none` (R14). If the silver labels hold,
  these are false accepts caused by classification, not extraction. They need a human check.
- 4 silver-`dividend` filings were classified `none` and went out of scope. They also need a
  check: either the model or the draft label is wrong.
- Silver labels are Claude drafts; numbers on them are indicative.

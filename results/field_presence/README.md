# Field presence in real filings

Runs of `python -m evals.analysis.field_presence`. Each folder is one run; runs are
never edited after the fact.

The question: before calling a field REQUIRED, how often does a real filing mention it at all?

## How to read the numbers

- **Mention, not disclosure.** A regex hit means the words appear somewhere in the text. Counts are
  upper bounds.
- **Dev split only.** The test split is not read, so design decisions don't leak into the test set.
- **Labels are mostly unverified drafts.** Each count is given for human-verified cases and for all
  cases. Only 2 verified dividend cases and 6 verified M&A cases exist in dev, so the "all" rows
  carry the signal and stay indicative.
- **Blind labelling stays blind.** `per_case.csv` has rows for human-verified cases only, because a
  row reveals the case's label (same rule as `results/e1`).
- **No LLM.** Running it costs nothing and is deterministic for a given dataset (see
  `manifest.json` for the dataset SHA-256 and git commit).

## Runs

| Folder | Cases | Notes |
|---|---|---|
| `20261112-202738_dev` | dev split: 20 dividend, 28 M&A (draft labels) | Current. `per_case.csv` lists human-verified cases only; `summary.json` counts all. |

## Spot check of `20261112-202738_dev` (by hand, 2026-11-12)

- `ex_dividend_date`: the only hit (E1-009) is an exception, not a routine disclosure. The dividend
  exceeded 25% of the share price, so the ex-dividend date followed a Nasdaq rule and the filing
  states it. In the other 19 dividend filings the term does not appear.
- `dollar_amount_million_billion` over-counts the deal value. Of the first 8 M&A filings, 3 contained
  the transaction value. 2 matched other amounts (a warrant, offering proceeds), and 3 had no match.

Consequence for the validator: before 2026-11-12, `ex_dividend_date`, `record_date` and
`payment_date` failed validation when empty, and `total_value_usd` failed when missing. That rule
would send a dividend filing without an ex-dividend date (19 of 20 here) into the retry loop. This
is inferred from the rule. No extraction was run.

# Human review → gold write-back (proposal, 2026-11-12)

Status: **proposal, not decided, not built.** The `review_queue` table exists (commit
`2b14d18`): documents a person must decide on are stored there with the reason, the
model's answer and the fields extracted so far. This note proposes how a reviewer's
decision should reach the gold `extractions` table. Chloe has not decided the open questions
below.

## Principle

What a person writes to gold passes **the same quality bar** as what the model writes. A
review does not bypass the rules silently.

## Three kinds of decision

| Decision | Example | What happens next |
|---|---|---|
| **Relabel** | `unclear` → actually M&A | Re-run extraction and validation with the human label (oracle doc type). Store if it passes; back to review if not |
| **Fill or correct fields** | the unnamed target is found in the text | Each value the reviewer enters carries a verbatim evidence quote, checked against the source like the classifier's quote. Then the same Extraction Validator runs; store if it passes |
| **Reject** | not a real event, or unusable input | Mark the review row resolved; nothing written to gold |

## Provenance in gold

- New `extractions` columns: `origin` (`model` / `human` / `model+human`) and `review_id`.
- The review row keeps a **field-level diff**: what the model said and what the person changed.
- A `human_review` record in `step_runs` keeps the audit trail complete and records how long a
  review took.

## Version conflicts

- For the same content (same `source_hash`), a later pipeline rerun does **not** overwrite a
  human-corrected version.
- New content (a new hash) gets a new model version; the human version is archived (SCD Type 2).

## Decisions as data

- Every decision is a labelled example, useful for error analysis (which field people keep
  fixing), for new scenario cases, and as material for E1/E2.
- Reviewed documents are the hard cases, a **biased sample**. Use them for error analysis
  and regression, not to estimate overall accuracy.
- New metrics: review precision (items sent to review that really needed a person), field
  edit rate, time per review.

## Implementation sketch

`apply_decision(review_id, decision)` reuses the validator and the repository. A command line
comes first (`python -m src.review resolve <id> --label mna --field target="..." --evidence "..."`),
a dashboard form later.

## Open questions (Chloe to decide)

1. May a person override a business rule (for example dates that really are unusual)?
   Proposal: yes, with a mandatory reason, and the override is flagged in gold.
2. Should the human version take precedence over later model reruns of the same content?
   Proposal: yes.
3. Should human decisions be used as evaluation data? Proposal: only for error analysis and
   regression, never for overall accuracy.
4. Which interface comes first? Proposal: CLI, because it is simpler and testable.

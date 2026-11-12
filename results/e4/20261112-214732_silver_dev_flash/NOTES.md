# E4 run 1: whole pipeline on the E1 dev split, silver labels (2026-11-12)

`deepseek-flash`, classifier input `head`, classify thinking off (the extractor uses the
provider default). Commit `cc92cd9`. 100 dev filings: 17 human-verified, 83 with draft
("silver") labels. 264,150 prompt + 39,718 completion tokens, 80 s at concurrency 5.

## Main finding: 4 of 42 supported filings were stored

| | Count |
|---|---|
| Gold M&A or Dividend (incl. multi-label) | 42 (dividend 17, mna 25) |
| ...stored | 3 (plus 1 gold `none`, see below) |
| ...human review at `validate` | 36 |
| Validate reviews caused by `TOTAL_VALUE_USD.POSITIVE` (value empty) | 26 |
| ...by `EX_DIVIDEND_DATE.DATE_FORMAT` (value empty) | 10 |
| ...by `RECORD_DATE.DATE_FORMAT` | 3 |

The validator still requires `total_value_usd` and `ex_dividend_date`, which the agreed
presence levels make OPTIONAL. The validator does not read the presence levels yet (feature
`presence_levels`, pending in the scenario gate). This confirms on real filings what the
keyword count found (`results/field_presence/`: ex-dividend date in 1 of 20 dividend filings).
**The next change is presence levels in the validator; then rerun E4.**

Every validate review used exactly one extraction: the retry policy held (no retry for
`missing_info`).

## Classification (E1 rules)

- Human-verified: 17/17 correct.
- Silver: 72/83 = 86.8 %. Silver labels are Claude drafts, so this is agreement with those
  drafts, not accuracy. Disagreements: gold dividend → none (4), none → mna (3),
  dividend+mna → mna (2), dividend+mna → none (1), none → unclear (1). They need human
  adjudication before anyone calls them errors.
- One gold-`none` filing was classified M&A and stored. If the silver label is right, that is a
  false accept; it needs checking.

## Not measured here

Field values (no field gold: see `evals/E2_DESIGN.md`).

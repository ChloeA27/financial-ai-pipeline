# Scenario set: synthetic CI gate, one case per pipeline path

Called the golden set until 2026-11-12. Renamed because a golden set usually means labelled,
representative data for measuring quality; this set is neither. The labelled sets are E1 (and later E2).

**Every input is synthetic.** The project author wrote them by hand as test fixtures. They
are not real SEC filings, even where they name real companies or describe a real event
(Microsoft / Activision Blizzard, Apple). Every other company, person and figure is
invented. Inputs added on 2026-11-12 carry the `syn_` prefix. Use these texts only to test
this pipeline.

The scenario set is a **regression gate**, not a quality measurement. It checks that a real
model, run end to end, still sends each kind of document down the right path. Accuracy on
real filings is measured by E1 (classification), and later E2 (extraction), on labelled
EDGAR data. That is why the gate needs one or two cases per path, not many. Each text is
written to leave the model no room for doubt, so that the gate does not flip between runs.

| Case | Input | Expected outcome | What it checks |
|---|---|---|---|
| S1 | `ma_microsoft_activision.txt` | stored, M&A | field extraction, all 8 fields |
| S2 | `dividend_apple_2026.txt` | stored, Dividend | field extraction, all 9 fields |
| S3 | `syn_ma_value_undisclosed.txt` | stored, M&A | deal value not disclosed: OPTIONAL field may stay empty |
| S4 | `syn_dividend_no_exdate.txt` | stored, Dividend | no ex-dividend date: OPTIONAL field may stay empty |
| O1 | `test_ma_fail_loop.txt` | out_of_scope | analyst speculation about a possible deal |
| O2 | `syn_mgmt_change_only.txt` | out_of_scope | officer change only |
| H1 | `syn_mna_and_dividend.txt` | human_review (classify / ambiguous) | an acquisition and a dividend in one release |
| H2 | `syn_ma_unnamed_target.txt` | human_review (validate / missing_info, `TARGET.NON_EMPTY`), 1 extraction | the target is not named; it must not be filled with a description, and there is no retry |
| H3 | `syn_dividend_bad_timeline.txt` | human_review (validate / business_rule), 1 extraction | record and ex-dividend dates fall before the declaration; no retry |
| D1 | `syn_binary_as_txt.txt` | dlq (read / input_error, `DECODE_FAILED`) | binary content stops at read, with no LLM call |

`test_ma_pass.txt` stays in `inputs/` because integration tests use it. It is not a scenario
case, because S1 already covers that path.

Paths that need a broken model or a broken API (output that never parses, a quote that
never matches, an outage) are covered by unit tests with a scripted fake LLM, not here.

Case format (`cases/<id>.json`): `case_id`, `input`, `synthetic`, `purpose`, `requires`
(features the case depends on, see `IMPLEMENTED_FEATURES` in `evals/evaluators/scenarios.py`), and
`expected` with `outcome`, optional `doc_type`, `failure` (`stage` / `category` / `code`), and,
for stored cases, `fields` and `field_overrides`.

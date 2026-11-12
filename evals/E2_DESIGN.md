# E2: extraction-stage evaluation (design, 2026-11-12)

Status: **design only**. No field-level gold exists yet, and no E2 number has been measured.

## Why

E1 measures whether a filing gets the right label. Nothing yet measures whether the fields
extracted from a correctly labelled filing are right. The scenario gate showed the main risk
on its first run (`results/scenarios/20261112-211858`, case H2): the model **filled a REQUIRED
field with a description it made up** after a feedback retry. The validator's checks
(non-empty, format, cross-field rules) cannot tell a real value from an invented one. Only a
comparison with labelled values can.

## Question

For each field of an M&A or Dividend filing, how often is the extracted value correct, wrong,
missing when the filing states it, or **invented when the filing does not state it**?

## Isolate the stage

E2 runs the Extractor and Extraction Validator with the **gold document type** (oracle
classification). A classification error therefore cannot show up as an extraction error. The
end-to-end effect of both stages together is measured separately: see E4 below.

## Dataset

- **Cases:** dev-split filings whose single supported label is `mna` or `dividend` (E1 dev: 20
  `mna`, 13 `dividend`, plus 12 with `mgmt_change` as a second label). Multi-label
  `mna;dividend` filings are left out until multi-label routing exists. The test split is
  untouched until the extraction prompt is final, as in E1.
- **Gold per field:** a value, **or an explicit `null` meaning "not stated in this filing"**.
  Without explicit nulls, fabrication cannot be measured. Every non-null value carries a
  verbatim evidence quote.
- **Labelling:** Claude pre-labels, then a human verifies each field against the quote. Status
  is `draft` until verified, and only verified cases count, as in E1. Record the labeller, the
  guide version and the date. Model answers on unverified cases stay sealed. The outsourced
  E1 sheet has no field values, so it cannot serve here.
- **Normalisation (labelling guide):** dates as ISO `YYYY-MM-DD`; amounts as numbers in the
  stated currency, with a currency field; percentages as 0–100. Names as written in the
  filing, with every spelling the filing uses listed as accepted (`one_of`).

## Metrics

Each (case, field) falls into one of five cells:

| | Model gives a value | Model gives null |
|---|---|---|
| **Gold has a value** | correct / **wrong** | **missed** |
| **Gold is null** | **fabricated** | correct null |

Per field, and pooled by presence level (REQUIRED / EXPECTED / OPTIONAL, from `ValidationProfile`):

- **Fabrication rate** = fabricated ÷ (gold null). This is the headline risk metric.
- **Miss rate** = missed ÷ (gold has a value).
- **Value accuracy** = correct ÷ (gold has a value).
- **Precision** = correct ÷ (model gives a value).
- **Record-level:** share of cases where every REQUIRED field is correct.

Routing metrics, using `outcome_of` on the same runs:

- **False accept rate:** stored records with a wrong or fabricated REQUIRED field ÷ stored records.
- **Review precision:** documents sent to human review that really needed a person (gold
  shows the value is missing or contradictory) ÷ documents sent to review.
- **Touchless rate:** stored and every REQUIRED field correct ÷ all cases.

Value comparison reuses the scenario gate's field types (`enum`, `numeric` ±1 %, `date`,
`text_lower`, `one_of`). Reported with counts, not only rates, because n is small.

## Protocol

- Same discipline as E1: frozen case list, temperature 0, thinking off, 2 repetitions, a
  manifest with model, prompt hash, dataset SHA-256 and git commit, and raw outputs in
  `results/e2/<run>/`.
- **One change per experiment against a baseline.** Planned arms: retry policy (done: no retry
  for `missing_info`), presence levels in the validator, strict structured output, and the
  extractor input window.
- Paired comparison on record-level correctness (McNemar, as in E1).
- Before scoring the model, the scorer is self-checked: gold as output gives 100 %, an
  all-null output gives fabrication 0 and recall 0, and a constant invented value is caught as
  fabricated wherever gold is null.

## Known limits to fix or measure first

- **The Extractor sees only the first 6,000 characters** (`src/nodes/extractor.py`). A value
  stated later in a long filing becomes a "miss" or a human review. Measure how often the gold
  evidence falls outside that window (coverage, no LLM), as E1 did for the classifier.
- Small n: about 30 cases means a 95 % interval of roughly ±15 points on any rate. Report
  counts with every rate.

## E4: end-to-end funnel (built 2026-11-12, see `evals/run_e4.py`)

The whole pipeline runs on the dev split, and each document's outcome (stored / out of scope /
human review / DLQ) is reported per gold label. Classification is scored against the labels,
human-verified and silver separately. Field values are **not** scored there; that is E2.

## Decisions needed from Chloe

1. Labelling effort: about 33 filings × 7–9 fields. Pre-label by Claude plus a human check is
   about 5 minutes per filing.
2. Should EXPECTED fields count toward the record-level metric?
3. Which E2 arm comes first after the baseline: the extractor window or presence levels?

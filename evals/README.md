# evals/ — probabilistic evaluation

`tests/` checks deterministic behaviour: no LLM calls, the same answer every run.
`evals/` measures how well the LLM pipeline performs on labelled documents. Results vary
by model, prompt and run, so every run records what it used.

```
evals/
├── datasets/
│   ├── golden/                 end-to-end gate (in repo, synthetic, one case per pipeline path)
│   │   ├── inputs/             source .txt files
│   │   └── cases/              <id>.json: expected outcome, failure stage/category, fields
│   └── e1/README.md            classification set: lives outside this repo (see there)
├── evaluators/
│   ├── golden.py               outcome gate + field gate (REQUIRED fields), pending, inconclusive
│   └── e1.py                   gold projection, input coverage, McNemar, repetition agreement
├── run_golden.py               runner: CI quality gate
└── run_e1.py                   runner: classification-stage experiments
```

Evaluators only score: they take a pipeline result plus a label and return a verdict.
Runners execute the pipeline and write the output. Production code in `src/` never imports
`evals/`, and a unit test enforces this.

## Golden set — the CI gate

```bash
python evals/run_golden.py [--json] [--no-save]
```

This runs the full pipeline on every case. Each case gets its own temporary database, raw
store and vector store. Each case states the outcome the pipeline must reach (`stored`,
`out_of_scope`, `human_review` or `dlq`) and, for a failure, its stage and category. There
are two gates, and they are never pooled:

- **Outcome gate:** every gating case must reach its expected outcome. One miss fails the
  gate, and a document stored when it should not be is reported as a false accept.
- **Field gate:** on correctly stored cases, every REQUIRED field (from the document type's
  `ValidationProfile`) must match. Other fields are reported as accuracy and warnings only.

A case that `requires` a feature that is not implemented yet runs and is reported as
*pending*, but it cannot fail the gate. An infrastructure failure (API down) is
*inconclusive*: the gate fails and the case is left out of every number. Exit code 1 means
the gate failed. Results are saved under `results/golden/<timestamp>/` unless `--no-save`
is given. CI runs it on
every push to `main` and on every pull request into it (`.github/workflows/ci.yml`). Run it
after changing a prompt, a schema or the model.

## E1 — classification stage

```bash
python evals/run_e1.py --coverage-only   # no LLM: is the gold evidence inside each window?
python evals/run_e1.py --estimate        # no LLM: planned calls and tokens
python evals/run_e1.py --reps 2 --max-calls 630
```

This compares classifier input strategies (V0 `head`, V1 `structured`, V2 `keyword`) as an
offline paired comparison. Each run writes `results/e1/<UTC timestamp>_<tag>/`, which holds
the manifest (dataset and input hashes, model, thinking mode, temperature, git commit), a
summary, per-case rows, and sealed answers for cases that are not yet human-verified. The
index of runs is in `results/e1/README.md`. The runner refuses the test split until the
prompt and input strategy are final (`--allow-test`).

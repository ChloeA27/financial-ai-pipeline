# evals/ — probabilistic evaluation

`tests/` checks deterministic behaviour: no LLM calls, the same answer every run.
`evals/` measures how well the LLM pipeline performs on labelled documents. Results vary
by model, prompt and run, so every run records what it used.

```
evals/
├── datasets/
│   ├── golden/                 end-to-end extraction set (in repo, 4 short sample announcements)
│   │   ├── inputs/             source .txt files
│   │   ├── expected_pass/      <name>.golden.json: fields the pipeline must extract
│   │   └── expected_fail/      <name>.golden.json: documents the pipeline must reject
│   └── e1/README.md            classification set: lives outside this repo (see there)
├── evaluators/
│   ├── golden.py               field-type-aware comparison, expected_fail scoring
│   └── e1.py                   gold projection, input coverage, McNemar, repetition agreement
├── run_golden.py               runner: CI quality gate
└── run_e1.py                   runner: classification-stage experiments
```

Evaluators only score: they take a pipeline result plus a label and return a verdict.
Runners execute the pipeline and write the output. Production code in `src/` never imports
`evals/`, and a unit test enforces this.

## Golden set — the CI gate

```bash
python evals/run_golden.py --threshold 0.85 [--json]
```

This runs the full pipeline on every input. Each case gets its own temporary database, raw
store and vector store. Exit code 1 means accuracy is below the threshold. CI runs it on
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

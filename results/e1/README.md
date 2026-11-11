# E1 runs: classification-stage evaluation

Each folder holds one run of `scripts/eval_classification.py`. The folder name is
`<UTC timestamp>_<tag>`. Runs are never edited or deleted after the fact. A run that
turned out to be wrong stays here with a `SUPERSEDED.md` note.

## Runs

| Folder | Mode | Model | Variants × reps | Cases | Status |
|---|---|---|---|---|---|
| `20261110-213902_coverage_orig35` | coverage only (no LLM) | — | 3 × — | 35 | **Superseded:** the first cover-page cut also removed Explanatory Notes. See its `SUPERSEDED.md`. |
| `20261110-214025_coverage_orig35` | coverage only (no LLM) | — | 3 × — | 35 | Current coverage numbers |
| `20261110-214813_smoke_flash` | run | `deepseek-flash` | 1 × 1 | 1 | Smoke test only. See its `NOTES.md`. |
| `20261110-214827_dev35_flash` | run | `deepseek-flash` | 3 × 2 | 35 | **Main result**, reported in `../SUMMARY.md`. See its `NOTES.md`. |

The `model` field in a coverage-only manifest is just the configured default. No model is
called in that mode.

## Folder layout

| File | Committed | Contents |
|---|---|---|
| `manifest.json` | yes | Dataset file name and SHA-256, split, guide versions, model, temperature, budget, variants, reps, git commit and dirty flag, and each case's `input_sha256` |
| `summary.json` | yes | Metrics per variant, by group, paired comparison (exact McNemar) |
| `per_case.csv` / `coverage.csv` | yes | One row per case × variant × rep. Cases that are not human-verified carry no labels or answers. |
| `_sealed/` | yes | Model answers on cases that are not yet human-verified. Do not open it before blind labelling is done. |
| `NOTES.md`, `SUPERSEDED.md` | yes | Human notes on the run |
| `run.db` | no (local) | SQLite: `documents` and `step_runs`. One row per node execution, with the raw LLM reply, tokens, latency and prompt hash. |
| `raw/<sha256>.txt` | no (local) | Bronze copies of the inputs. Each file name is the SHA-256 of the file's bytes. |

Everything is linked by hash. `manifest.json` → `input_sha256` → `raw/<sha256>.txt`.
In `run.db`, `step_runs.source_hash` gives the same hash. To check a copy, recompute
the SHA-256 of a file in `raw/` (or of a dataset input) and compare it with the manifest.
`run.db` and `raw/` stay out of git because of their size (one input is 57 MB). Back them up
privately.

Browse a run with `streamlit run src/e1_dashboard.py` (localhost only).

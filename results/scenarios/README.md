# Scenario gate runs

One folder per local run of `python evals/run_scenarios.py` (CI runs with `--no-save`). Folders
are never edited after the fact; human notes go in `NOTES.md`.

| Folder | Model | Gate | Notes |
|---|---|---|---|
| `20261112-211858` | `deepseek-chat` | FAIL (H2 false accept; S1 test-data bug) | First run. See `NOTES.md` |
| `20261112-213217` | `deepseek-chat` | PASS (8/8; S3, S4 pending) | After the retry policy. H2 fixed. 15 vs 26 LLM calls. See `NOTES.md` |
| `20261112-215520` | `deepseek-chat` | PASS (10/10) | After presence levels: S3, S4 now gate and pass |

# Scenario gate runs

One folder per local run of `python evals/run_scenarios.py` (CI runs with `--no-save`). Folders
are never edited after the fact; human notes go in `NOTES.md`.

| Folder | Model | Gate | Notes |
|---|---|---|---|
| `20261112-211858` | `deepseek-chat` | FAIL (H2 false accept; S1 test-data bug) | First run. See `NOTES.md` |

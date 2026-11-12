# E4 runs: whole pipeline on the E1 dev split

One folder per run of `python -m evals.run_e4`. View with `make dash`. Folders are never
edited after the fact; notes go in `NOTES.md`. `run.db`, `raw/` and `_local/` stay local
(`_local/` holds labels and answers for unverified cases).

| Folder | Model | Stored / review / out of scope / DLQ | Notes |
|---|---|---|---|
| `20261112-214732_silver_dev_flash` | `deepseek-flash` | 4 / 40 / 56 / 0 | Validator still requires two OPTIONAL fields; see `NOTES.md` |
| `20261112-215526_silver_dev_flash_presence` | `deepseek-flash` | 40 / 4 / 56 / 0 | After presence levels: 35 of 42 supported filings stored (run 1: 2). See `NOTES.md` |

# E1 dataset — kept outside this repository

The E1 classification set has SEC EDGAR 8-K / 6-K filings and their labels. It is not
committed here, for two reasons:

- **Blind labelling.** The draft labels of cases that are not yet human-verified must not
  be visible while they are being labelled. This repository is public.
- **Size.** The inputs are about 77 MB and the full EDGAR submissions about 825 MB. One
  input alone is 57 MB.

Point the runner at your local copy once, in `.env` (git-ignored):

```
E1_DATASET=/path/to/e1_classification/e1_dataset_draft.jsonl
```

Inputs are read from `<dataset dir>/inputs/` and are only read, never written. Every run
records the dataset's SHA-256 and each input's SHA-256 in `results/e1/<run>/manifest.json`,
so a result can be tied back to the exact files without publishing them.

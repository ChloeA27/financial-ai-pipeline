# Golden set — synthetic CI smoke test

**All four inputs are synthetic.** The project author wrote them by hand as test fixtures.
They are not real SEC filings, even where they name real companies or describe a real
event (Microsoft / Activision Blizzard). The analyst note in `test_ma_fail_loop.txt` is
invented, including the author and the firm it names. Do not use these texts for anything
but testing this pipeline.

| Input | Case | What it checks |
|---|---|---|
| `test_ma_pass.txt` | expected_pass | M&A extraction, all 8 fields |
| `ma_microsoft_activision.txt` | expected_pass | M&A extraction, all 8 fields |
| `dividend_apple_2026.txt` | expected_pass | Dividend extraction, all 9 fields |
| `test_ma_fail_loop.txt` | expected_fail | a speculative analyst note must be rejected, not stored |

The set is a regression smoke test run by CI (`evals/run_golden.py`), not an evaluation
set. It has 4 cases, no dev/test split and one labeller. Measured extraction quality needs
an E2 set built from real filings (see `evals/README.md`).

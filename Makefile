# Shortcuts. Uses the project venv; `python -m` because the venv's console
# scripts point at an old path.
PY := .venv/bin/python

.PHONY: dash test scenarios e4-estimate

dash:  ## all dashboards (E4, E1, pipeline DB) at http://localhost:8501
	@echo "Dashboards: http://localhost:8501  (Ctrl+C to stop)"
	@(sleep 3; open http://localhost:8501 2>/dev/null || true) &
	$(PY) -m streamlit run src/app.py --server.headless true

test:  ## offline unit tests (no LLM calls)
	$(PY) -m pytest -m "not integration" -q

scenarios:  ## scenario gate with the real LLM (costs tokens)
	$(PY) evals/run_scenarios.py

e4-estimate:  ## E4 case count and token estimate, no LLM
	$(PY) -m evals.run_e4 --estimate

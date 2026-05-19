.PHONY: help test test-contract test-fast test-llm install build-index lint

PY ?= python3
VENV ?= .venv
VENV_PY := $(VENV)/bin/python

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Create .venv and install requirements
	$(PY) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

build-index:  ## Build the RAG vector index (issue #7)
	$(VENV_PY) -m agent.rag.build_index

test-contract:  ## Run the schema-conformance canary (issue #2). Required to pass on every commit.
	$(VENV_PY) tests/test_contract.py

test-fast:  ## Run every test that doesn't need OPENAI_API_KEY. Should be < 30s on a laptop.
	@for f in tests/test_audit.py tests/test_buildings.py tests/test_clarify.py tests/test_contract.py \
	          tests/test_hitl.py tests/test_profiles.py tests/test_risk.py tests/test_risk_assignment.py \
	          tests/test_vendors.py; do \
		echo "=== $$f ==="; $(VENV_PY) $$f || exit 1; \
	done

test-llm:  ## Run the real-OpenAI E2E tests. Requires OPENAI_API_KEY in .env.
	@for f in tests/test_extract.py tests/test_classify.py; do \
		echo "=== $$f ==="; $(VENV_PY) $$f || exit 1; \
	done

test: test-fast  ## Default `make test` runs the fast suite (CI gate). LLM tests via `make test-llm`.

lint:  ## (placeholder) Add ruff / mypy here if/when adopted.
	@echo "no linter configured yet"

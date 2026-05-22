.PHONY: help test test-contract test-fast test-llm test-backend install build-index lint demo backend-only frontend-only

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
	@for f in tests/test_*.py; do \
		case "$$(basename $$f)" in \
		  test_classify.py|test_extract.py|test_rag_build.py|test_rag_retriever.py) continue ;; \
		esac; \
		echo "=== $$f ==="; $(VENV_PY) $$f || exit 1; \
	done

test-llm:  ## Run the real-OpenAI E2E tests. Requires OPENAI_API_KEY in .env.
	@for f in tests/test_classify.py tests/test_extract.py tests/test_rag_build.py tests/test_rag_retriever.py; do \
		echo "=== $$f ==="; $(VENV_PY) $$f || exit 1; \
	done

test: test-fast  ## Default `make test` runs the fast suite (CI gate). LLM tests via `make test-llm`.

lint:  ## (placeholder) Add ruff / mypy here if/when adopted.
	@echo "no linter configured yet"

# ─── Live demo targets ────────────────────────────────────────────────

test-backend:  ## Run the backend route + classify_with_events tests
	$(VENV_PY) -m pytest tests/test_backend_routes.py tests/test_classify_with_events.py tests/test_classify_unchanged.py -q

backend-only:  ## Start FastAPI on :8000 (demo backend only)
	$(VENV_PY) -m uvicorn backend.main:app --reload --port 8000

frontend-only:  ## Start Next.js on :3000 (issue-demo-frontend)
	@if [ -d frontend ]; then \
		cd frontend && (npm install --silent && npm run dev); \
	else \
		echo "frontend/ does not exist yet"; exit 1; \
	fi

demo:  ## Start backend (:8000) and frontend (:3000) together; Ctrl-C tears both down
	@echo ">>> starting CBRE HITL live demo (backend :8000, frontend :3000)"
	@trap 'echo; echo ">>> stopping demo"; kill 0' INT TERM EXIT; \
	$(VENV_PY) -m uvicorn backend.main:app --port 8000 & \
	if [ -d frontend ]; then \
		(cd frontend && npm run dev) & \
	else \
		echo ">>> frontend/ not found, running backend-only"; \
	fi; \
	wait

PY = ../.venv/bin/python
PYTEST = ../.venv/bin/python -m pytest
RUFF = ../.venv/bin/ruff

.PHONY: api web test lint eval record install docker

install:
	uv pip install --python .venv/bin/python -e backend
	uv pip install --python .venv/bin/python pytest pytest-asyncio httpx ruff
	cd frontend && npm install

api:
	cd backend && $(PY) -m uvicorn graph_rag.api.main:app --reload --port 8000

web:
	cd frontend && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev

test:
	cd backend && $(PYTEST) -q

lint:
	cd backend && $(RUFF) check graph_rag tests

eval:
	$(PY) backend/evals/run.py

record:
	set -a; . ./.env; set +a; $(PY) backend/evals/record.py

build:
	cd frontend && npm run build

docker:
	docker compose up --build

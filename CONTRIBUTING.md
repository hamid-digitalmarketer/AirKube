# Contributing to AirKube

Thanks for your interest in contributing! AirKube is a LangGraph agent + Streamlit
dashboard wired to a Neo4j knowledge graph, a FastAPI inference service, and
Airflow/BigQuery data pipelines. This guide covers how to get set up and how to
submit changes.

## Good first issues

If you're new to the project, start with issues labeled
[`good first issue`](https://github.com/iamvisheshsrivastava/AirKube/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
These are small, self-contained, and don't require a full Neo4j/Airflow stack to
verify — usually just `pytest`.

## Dev setup

```bash
git clone https://github.com/iamvisheshsrivastava/AirKube.git
cd AirKube
python -m venv .venv
.venv\Scripts\activate        # Windows (use `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements-dev.txt   # installs requirements.txt + pytest
```

`requirements-dev.txt` layers on top of `requirements.txt`, so installing it is
enough for both running the app and running tests.

### Environment variables

Copy `.env.example` to `.env` and fill in what you need:

```bash
cp .env.example .env
```

You don't need every variable to work on most issues:

| Variable(s) | Needed for |
|---|---|
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | Running the LangGraph agent / dashboard chat / extraction playground |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | KG Explorer, `query_knowledge_graph`/`record_deployment` agent tools, `ml/kg_*.py`. Without these, `ml/kg_utils.py` degrades to a "simulation mode" that logs Cypher instead of running it — fine for most non-KG work |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Pipeline-trigger/status agent tools and `check_system_health`. Without all three, those tools honestly report "not configured" rather than faking success |
| `NEWS_API_KEY`, `GCP_PROJECT_ID`, `NEWS_GCS_BUCKET` | The news ETL pipeline (`dags/news_data_pipeline.py`, `ml/news_*.py`) |
| `API_KEY` | Optional shared secret for the inference API (`ml/inference.py`). Leave unset for local dev (auth disabled) |
| `DD_API_KEY`, `DD_APP_KEY`, `DD_SITE` | Datadog APM — only relevant in production; safe to leave unset locally |
| `AIRKUBE_CHECKPOINT_DB` | LangGraph conversation checkpointing (defaults to `data/checkpoints.sqlite`; set to `:memory:` to disable persistence) |

See `.env.example` for the full list and `README.md`'s "Environment Variables"
table for descriptions.

### Running the app locally

```bash
streamlit run dashboard.py          # dashboard → http://localhost:8501
uvicorn ml.inference:app --reload   # inference API → http://localhost:8000
python run_agent.py                 # CLI agent
```

For the full stack including Airflow and Neo4j, use Docker Compose (set
`NEO4J_AUTH`/`NEO4J_PASSWORD` in `.env` first):

```bash
docker compose up
```

## Running tests

```bash
pytest tests/ -v
```

Tests live in `tests/` and are organized by the module they cover
(`test_agent_tools.py`, `test_inference.py`, `test_kg_ingestion.py`,
`test_news_pipeline.py`). Most tests mock external services (Neo4j, Airflow,
OpenRouter) with `unittest.mock`/`monkeypatch` rather than requiring them to be
running — follow that pattern for new tests so they can run in CI (see
`.github/workflows/tests.yml`, which runs `pytest tests/ -v` on every push/PR to
`main`).

If you're fixing a bug, add a regression test for it in the relevant
`tests/test_*.py` file, following the existing "Regression test for #N" comment
style used throughout the suite.

## Code style

This repo doesn't currently enforce a formatter/linter in CI, so keep changes
consistent with the surrounding code:

- Match the existing style in the file you're editing (naming, docstring
  format, logging via the module-level `logger`, f-strings for messages).
- Favor explicit, honest error handling: several past bugs in this codebase
  were "hardcoded/faked success" responses. New code (especially agent tools
  and health/status checks) should report failures and unconfigured states
  honestly rather than fabricating a success result — this is treated as a
  hard requirement, not a style preference.
- Add docstrings to new public functions/tools, especially `@tool`-decorated
  agent functions in `agent/tools.py`, since their docstrings are what the LLM
  sees.
- Never interpolate untrusted input (LLM output, user input) directly into
  Cypher strings — use parameterized queries (see `ml/kg_utils.py`'s
  `run_query`/`run_write_query`) or an explicit allowlist (see
  `ml/kg_ingestion.py`'s `safe_rel_type`).

## Pull request process

1. Fork the repo and create a branch off `main` (`git checkout -b fix/short-description`).
2. Make your change, with tests covering it.
3. Run `pytest tests/ -v` locally and make sure everything passes.
4. Open a PR against `main` with:
   - A short description of what changed and why.
   - The issue number it fixes/addresses (e.g. "Fixes #23"), if applicable.
   - Confirmation that tests pass.
5. Keep PRs focused — one issue/fix per PR makes review much faster.
6. Be responsive to review comments; small follow-up commits are fine (no need
   to force-push/squash until a maintainer asks).

## Reporting bugs / proposing features

Open a GitHub issue. For bugs, include: what you expected, what happened
instead, and repro steps (including which env vars were/weren't set, since a
lot of behavior here depends on that). For features, a short description of
the use case is enough to start a discussion.

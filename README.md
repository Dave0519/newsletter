# newsletter

Python newsletter workspace with two execution paths:

- `clue_letter/`: service-style user workflow and HTML generation
- `openclaw/`: staged batch pipeline and operator-facing observability

## Openclaw Read-Only Observability API

This repo now includes a lightweight read-only API for inspecting `openclaw` runtime ledgers without a web UI.

Start the server from the repo root:

```bash
python3 openclaw/api_server.py --host 127.0.0.1 --port 8765
```

Available endpoints:

- `/health`
- `/api/summary`
- `/api/runs`
- `/api/stages`
- `/api/losses`
- `/api/customers`
- `/api/sources`
- `/api/cron`
- `/api/checkpoints?run_id=<RUN_ID>&customer_id=<CUSTOMER_ID>&include_data=1`

Example calls:

```bash
curl "http://127.0.0.1:8765/api/summary"
curl "http://127.0.0.1:8765/api/stages?run_id=<RUN_ID>"
curl "http://127.0.0.1:8765/api/losses?run_id=<RUN_ID>&customer_id=<CUSTOMER_ID>"
curl "http://127.0.0.1:8765/api/checkpoints?run_id=<RUN_ID>&customer_id=<CUSTOMER_ID>&include_data=1"
```

## Runtime Ledgers

`openclaw` continues to use the existing runtime contracts and now emits additional additive ledgers at the repo root:

- `newsletter_agent_state.json`
- `newsletter_runs.jsonl`
- `newsletter_audit.jsonl`
- `newsletter_stage_runs.jsonl`
- `newsletter_loss_events.jsonl`
- `newsletter_customer_decisions.jsonl`
- `newsletter_source_health.jsonl`

These files are read-only operator surfaces. They explain:

- which stage ran
- where items dropped out
- how many items survived each stage
- what each customer finally received
- which sources degraded across shortlist/precheck/process/select steps

## Verification

Targeted tests for the new observability layer:

```bash
python3 -m unittest discover -s tests/openclaw -p "test_observability_*.py"
```

Manual API smoke test:

```bash
python3 openclaw/api_server.py --host 127.0.0.1 --port 8765
curl "http://127.0.0.1:8765/api/summary"
```

## Existing Service Flow

The service-side user workflow remains documented in `clue_letter/README.md`.

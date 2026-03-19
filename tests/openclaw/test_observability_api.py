from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openclaw.agents.observability import ObservabilityStore
from openclaw.api_server import build_payload


class ObservabilityApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "openclaw" / "cron").mkdir(parents=True, exist_ok=True)
        (self.root / "openclaw" / "cron" / "jobs.json").write_text(json.dumps({"jobs": []}), encoding="utf-8")
        self.store = ObservabilityStore(str(self.root / "openclaw"))

        state = {
            "run_id": "run-1",
            "stage": "F",
            "paths": {
                "master_candidate_pool": str(self.root / "pool.json"),
                "shortlists": {"c1": str(self.root / "shortlist.json")},
            },
        }
        self.store.state_path().write_text(json.dumps(state), encoding="utf-8")
        (self.root / "pool.json").write_text(json.dumps([{"title": "pool"}]), encoding="utf-8")
        (self.root / "shortlist.json").write_text(json.dumps([{"title": "short"}]), encoding="utf-8")

        self.store.append_stage(
            run_id="run-1",
            stage_id="B",
            status="completed",
            in_count=10,
            out_count=4,
            loss_count=6,
            customer_id="c1",
        )
        self.store.append_loss_events(
            run_id="run-1",
            stage_id="C",
            reason_code="precheck_rejected",
            customer_id="c1",
            items=[{"title": "drop", "url": "https://example.com/a", "source": "rss"}],
        )
        self.store.append_customer_decision(
            run_id="run-1",
            customer_id="c1",
            coverage={"need_gap": False},
            selected=[{"title": "sel", "url": "https://example.com/b", "source": "rss"}],
            gate_ok=True,
            delivery={"email": True},
            soft_warnings=[],
            gate_errors=[],
        )
        self.store.append_source_health(
            run_id="run-1",
            customer_id="c1",
            snapshot={"sources": [{"domain": "example.com", "counts": {"selected": 1}}]},
        )
        self.store.append_jsonl(self.store.runs_path(), {"runId": "run-1", "customerId": "c1", "status": "ok"})

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_summary_contains_linked_ledgers(self) -> None:
        status, payload = build_payload(self.store, "/api/summary", {"run_id": ["run-1"]})
        self.assertEqual(status, 200)
        self.assertEqual(payload["runId"], "run-1")
        self.assertEqual(len(payload["stages"]), 1)
        self.assertEqual(len(payload["losses"]), 1)
        self.assertEqual(len(payload["customerDecisions"]), 1)

    def test_checkpoints_can_include_data(self) -> None:
        status, payload = build_payload(
            self.store,
            "/api/checkpoints",
            {"run_id": ["run-1"], "customer_id": ["c1"], "include_data": ["1"]},
        )
        self.assertEqual(status, 200)
        self.assertIn("shortlists", payload["paths"])
        self.assertEqual(payload["paths"]["shortlists"]["c1"]["data"][0]["title"], "short")

    def test_losses_filter_by_customer(self) -> None:
        status, payload = build_payload(self.store, "/api/losses", {"customer_id": ["c1"]})
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["reasonCode"], "precheck_rejected")


if __name__ == "__main__":
    unittest.main()

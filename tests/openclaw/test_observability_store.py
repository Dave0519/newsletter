from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openclaw.agents.observability import ObservabilityStore, candidate_id, diff_candidates


class ObservabilityStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "openclaw" / "cron").mkdir(parents=True, exist_ok=True)
        self.store = ObservabilityStore(str(root / "openclaw"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_diff_candidates_tracks_url_identity(self) -> None:
        before = [
            {"title": "one", "url": "https://example.com/a"},
            {"title": "two", "url": "https://example.com/b"},
        ]
        after = [{"title": "one", "url": "https://example.com/a"}]
        dropped = diff_candidates(before, after)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["title"], "two")

    def test_source_health_snapshot_shows_stage_drops(self) -> None:
        collected = [
            {"title": "one", "url": "https://example.com/a", "source": "rss"},
            {"title": "two", "url": "https://example.com/b", "source": "rss"},
        ]
        shortlisted = collected[:1]
        prechecked = shortlisted[:1]
        cleaned = []
        selected = []
        snapshot = self.store.build_source_health_snapshot(
            collected=collected,
            shortlisted=shortlisted,
            prechecked=prechecked,
            cleaned=cleaned,
            selected=selected,
        )
        self.assertEqual(len(snapshot["sources"]), 1)
        first = snapshot["sources"][0]
        self.assertEqual(first["counts"]["collected"], 2)
        self.assertIn("losses", first)

    def test_stage_and_loss_ledgers_append(self) -> None:
        item = {"title": "lost", "url": "https://example.com/lost", "source": "rss"}
        self.store.append_stage(
            run_id="run-1",
            stage_id="B",
            status="completed",
            in_count=2,
            out_count=1,
            loss_count=1,
            customer_id="c1",
        )
        self.store.append_loss_events(
            run_id="run-1",
            stage_id="B",
            reason_code="not_shortlisted",
            items=[item],
            customer_id="c1",
        )
        stages = self.store.read_stages()
        losses = self.store.read_losses()
        self.assertEqual(stages[-1]["runId"], "run-1")
        self.assertEqual(losses[-1]["candidateId"], candidate_id(item))


if __name__ == "__main__":
    unittest.main()

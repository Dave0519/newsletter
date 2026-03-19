from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


SEOUL = ZoneInfo("Asia/Seoul")


def now_iso() -> str:
    return datetime.now(SEOUL).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    if limit is not None:
        return rows[-limit:]
    return rows


def candidate_id(article: dict[str, Any]) -> str:
    url = (article.get("url") or article.get("source_url") or "").strip().lower()
    title = (article.get("title") or article.get("title_from_url") or "").strip().lower()
    published = (article.get("published_at") or "").strip().lower()
    base = url or f"{title}|{published}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def article_source_meta(article: dict[str, Any]) -> dict[str, Any]:
    raw_url = (article.get("url") or article.get("source_url") or "").strip()
    host = ""
    if raw_url:
        try:
            host = (urlparse(raw_url).netloc or "").lower()
        except Exception:
            host = ""
    if host.startswith("www."):
        host = host[4:]
    return {
        "source": article.get("source", ""),
        "sourceCategory": article.get("source_category", article.get("category", "")),
        "domain": host,
    }


def diff_candidates(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    after_ids = {candidate_id(item) for item in after}
    return [item for item in before if candidate_id(item) not in after_ids]


class ObservabilityStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.root_dir = self.base_dir.parent

    def _path(self, name: str) -> Path:
        return self.root_dir / name

    def state_path(self) -> Path:
        return self._path("newsletter_agent_state.json")

    def runs_path(self) -> Path:
        return self._path("newsletter_runs.jsonl")

    def audit_path(self) -> Path:
        return self._path("newsletter_audit.jsonl")

    def stages_path(self) -> Path:
        return self._path("newsletter_stage_runs.jsonl")

    def losses_path(self) -> Path:
        return self._path("newsletter_loss_events.jsonl")

    def customer_decisions_path(self) -> Path:
        return self._path("newsletter_customer_decisions.jsonl")

    def source_health_path(self) -> Path:
        return self._path("newsletter_source_health.jsonl")

    def append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_stage(
        self,
        *,
        run_id: str,
        stage_id: str,
        status: str,
        in_count: int,
        out_count: int,
        loss_count: int,
        customer_id: str | None = None,
        checkpoint_path: str = "",
        metadata: dict[str, Any] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        payload = {
            "timestamp": now_iso(),
            "runId": run_id,
            "stageId": stage_id,
            "status": status,
            "customerId": customer_id,
            "inCount": int(in_count),
            "outCount": int(out_count),
            "lossCount": int(loss_count),
            "checkpointPath": checkpoint_path,
            "errors": errors or [],
            "metadata": metadata or {},
        }
        self.append_jsonl(self.stages_path(), payload)

    def append_loss_events(
        self,
        *,
        run_id: str,
        stage_id: str,
        reason_code: str,
        items: list[dict[str, Any]],
        customer_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for item in items:
            source_meta = article_source_meta(item)
            payload = {
                "timestamp": now_iso(),
                "runId": run_id,
                "stageId": stage_id,
                "customerId": customer_id,
                "reasonCode": reason_code,
                "candidateId": candidate_id(item),
                "title": item.get("title") or item.get("title_from_url") or "",
                "url": item.get("url") or item.get("source_url") or "",
                "metadata": metadata or {},
                **source_meta,
            }
            self.append_jsonl(self.losses_path(), payload)

    def append_customer_decision(
        self,
        *,
        run_id: str,
        customer_id: str,
        coverage: dict[str, Any],
        selected: list[dict[str, Any]],
        gate_ok: bool,
        delivery: dict[str, Any],
        soft_warnings: list[str],
        gate_errors: list[str],
    ) -> None:
        selected_items = []
        for item in selected:
            source_meta = article_source_meta(item)
            selected_items.append(
                {
                    "candidateId": candidate_id(item),
                    "title": item.get("title") or item.get("title_from_url") or "",
                    "url": item.get("url") or item.get("source_url") or "",
                    "country": item.get("country", ""),
                    "category": item.get("category", ""),
                    "customerScore": item.get("_customer_score", 0),
                    **source_meta,
                }
            )
        payload = {
            "timestamp": now_iso(),
            "runId": run_id,
            "customerId": customer_id,
            "gateOk": gate_ok,
            "delivery": delivery,
            "softWarnings": soft_warnings,
            "gateErrors": gate_errors,
            "coverage": coverage,
            "selected": selected_items,
        }
        self.append_jsonl(self.customer_decisions_path(), payload)

    def append_source_health(
        self,
        *,
        run_id: str,
        customer_id: str,
        snapshot: dict[str, Any],
    ) -> None:
        payload = {
            "timestamp": now_iso(),
            "runId": run_id,
            "customerId": customer_id,
            **snapshot,
        }
        self.append_jsonl(self.source_health_path(), payload)

    def build_source_health_snapshot(
        self,
        *,
        collected: list[dict[str, Any]],
        shortlisted: list[dict[str, Any]],
        prechecked: list[dict[str, Any]],
        cleaned: list[dict[str, Any]],
        selected: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stage_sets = {
            "collected": {candidate_id(item): item for item in collected},
            "shortlisted": {candidate_id(item): item for item in shortlisted},
            "prechecked": {candidate_id(item): item for item in prechecked},
            "cleaned": {candidate_id(item): item for item in cleaned},
            "selected": {candidate_id(item): item for item in selected},
        }
        rows: dict[str, dict[str, Any]] = {}
        for stage_name, items in stage_sets.items():
            for cid, item in items.items():
                meta = article_source_meta(item)
                row_key = f"{meta['domain']}|{meta['source']}|{meta['sourceCategory']}"
                row = rows.setdefault(
                    row_key,
                    {
                        "domain": meta["domain"],
                        "source": meta["source"],
                        "sourceCategory": meta["sourceCategory"],
                        "counts": {
                            "collected": 0,
                            "shortlisted": 0,
                            "prechecked": 0,
                            "cleaned": 0,
                            "selected": 0,
                        },
                        "sampleTitles": [],
                    },
                )
                row["counts"][stage_name] += 1
                title = item.get("title") or item.get("title_from_url") or ""
                if title and len(row["sampleTitles"]) < 3 and title not in row["sampleTitles"]:
                    row["sampleTitles"].append(title)
        for row in rows.values():
            counts = row["counts"]
            row["losses"] = {
                "shortlistDrop": counts["collected"] - counts["shortlisted"],
                "precheckDrop": counts["shortlisted"] - counts["prechecked"],
                "processingDrop": counts["prechecked"] - counts["cleaned"],
                "selectionDrop": counts["cleaned"] - counts["selected"],
            }
        return {"sources": sorted(rows.values(), key=lambda row: (row["domain"], row["source"]))}

    def read_state(self) -> dict[str, Any]:
        return _read_json(self.state_path(), {})

    def read_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return _read_jsonl(self.runs_path(), limit=limit)

    def read_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        return _read_jsonl(self.audit_path(), limit=limit)

    def read_stages(self, limit: int = 200) -> list[dict[str, Any]]:
        return _read_jsonl(self.stages_path(), limit=limit)

    def read_losses(self, limit: int = 200) -> list[dict[str, Any]]:
        return _read_jsonl(self.losses_path(), limit=limit)

    def read_customer_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        return _read_jsonl(self.customer_decisions_path(), limit=limit)

    def read_source_health(self, limit: int = 50) -> list[dict[str, Any]]:
        return _read_jsonl(self.source_health_path(), limit=limit)

    def read_cron_state(self) -> dict[str, Any]:
        return _read_json(self.base_dir / "cron" / "jobs.json", {})

    def checkpoints_for_run(self, run_id: str | None = None) -> dict[str, Any]:
        state = self.read_state()
        if not run_id:
            run_id = state.get("run_id")
        if not run_id or state.get("run_id") != run_id:
            return {}
        paths = state.get("paths", {}) if isinstance(state, dict) else {}
        return {
            "runId": run_id,
            "paths": paths,
        }

    def run_snapshot(self, run_id: str | None = None) -> dict[str, Any]:
        state = self.read_state()
        current_run_id = run_id or state.get("run_id")

        def _filter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not current_run_id:
                return rows
            return [row for row in rows if row.get("runId") == current_run_id]

        stages = _filter(self.read_stages(limit=500))
        losses = _filter(self.read_losses(limit=500))
        customer_decisions = _filter(self.read_customer_decisions(limit=100))
        source_health = _filter(self.read_source_health(limit=100))
        runs = _filter(self.read_runs(limit=100))
        return {
            "runId": current_run_id,
            "state": state if not current_run_id or state.get("run_id") == current_run_id else {},
            "runs": runs,
            "stages": stages,
            "losses": losses,
            "customerDecisions": customer_decisions,
            "sourceHealth": source_health,
            "checkpoints": self.checkpoints_for_run(current_run_id),
            "cron": self.read_cron_state(),
        }

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from agents.observability import ObservabilityStore
except ImportError:
    from openclaw.agents.observability import ObservabilityStore


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _limit_from_query(query: dict[str, list[str]], default: int = 50) -> int:
    try:
        value = int(query.get("limit", [str(default)])[0])
    except Exception:
        value = default
    return max(1, min(value, 500))


def _filter_rows(rows: list[dict[str, Any]], query: dict[str, list[str]]) -> list[dict[str, Any]]:
    run_id = query.get("run_id", [""])[0]
    customer_id = query.get("customer_id", [""])[0]
    stage_id = query.get("stage_id", [""])[0]

    filtered = rows
    if run_id:
        filtered = [row for row in filtered if row.get("runId") == run_id]
    if customer_id:
        filtered = [row for row in filtered if row.get("customerId") == customer_id]
    if stage_id:
        filtered = [row for row in filtered if row.get("stageId") == stage_id]
    return filtered


def _checkpoints_payload(store: ObservabilityStore, query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = query.get("run_id", [""])[0] or None
    customer_id = query.get("customer_id", [""])[0] or None
    include_data = query.get("include_data", ["0"])[0] == "1"
    payload = store.checkpoints_for_run(run_id)
    if not include_data or not payload:
        return payload

    paths = payload.get("paths", {})
    out_paths: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, str):
            out_paths[key] = {"path": value, "data": _read_json(Path(value))}
            continue
        if isinstance(value, dict):
            selected: dict[str, Any] = {}
            for child_key, child_value in value.items():
                if customer_id and child_key != customer_id:
                    continue
                selected[child_key] = {"path": child_value, "data": _read_json(Path(child_value))}
            out_paths[key] = selected
            continue
        out_paths[key] = value
    return {"runId": payload.get("runId"), "paths": out_paths}


def build_payload(store: ObservabilityStore, path: str, query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    limit = _limit_from_query(query)

    if path == "/health":
        return 200, {"ok": True}
    if path == "/api/summary":
        run_id = query.get("run_id", [""])[0] or None
        return 200, store.run_snapshot(run_id)
    if path == "/api/runs":
        return 200, {"items": _filter_rows(store.read_runs(limit=limit), query)}
    if path == "/api/stages":
        return 200, {"items": _filter_rows(store.read_stages(limit=limit), query)}
    if path == "/api/losses":
        return 200, {"items": _filter_rows(store.read_losses(limit=limit), query)}
    if path == "/api/customers":
        return 200, {"items": _filter_rows(store.read_customer_decisions(limit=limit), query)}
    if path == "/api/sources":
        return 200, {"items": _filter_rows(store.read_source_health(limit=limit), query)}
    if path == "/api/cron":
        return 200, store.read_cron_state()
    if path == "/api/checkpoints":
        return 200, _checkpoints_payload(store, query)

    return 404, {"ok": False, "error": "not_found", "path": path}


def make_handler(project_root: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            store = ObservabilityStore(str(Path(project_root) / "openclaw"))
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            status, payload = build_payload(store, parsed.path, query)
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return None

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only API for openclaw observability ledgers")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="project root path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.root))
    print(json.dumps({"ok": True, "host": args.host, "port": args.port}, ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

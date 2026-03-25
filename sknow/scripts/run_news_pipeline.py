#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import logging
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def _load_env_file(path: Path, env: dict[str, str]) -> None:
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and k not in env:
                env[k] = v
    except Exception:
        return


def _load_pipeline_env() -> dict[str, str]:
    env = os.environ.copy()
    for path in [
        Path("/Users/davechoi/.openclaw/.env"),
        Path("/Users/davechoi/.openclaw/workspace/openclaw/.env"),
    ]:
        _load_env_file(path, env)
    env.setdefault("OPENAI_MODEL", "gpt-5-mini")
    return env




def _parse_py_version(raw: str) -> tuple[int, int, int]:
    parts = []
    for token in raw.strip().replace("-", ".").split(".")[:3]:
        try:
            parts.append(int(token))
        except Exception:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _candidate_python_paths() -> list[str]:
    return [
        os.environ.get("SKNOW_PYTHON"),
        "/usr/local/bin/python3.12",
        "/usr/bin/python3.12",
        "/usr/local/bin/python3.11",
        "/usr/bin/python3.11",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
        shutil.which("python3"),
        shutil.which("python"),
    ]


def _find_python(preferred: str | None = None, min_version: tuple[int, int, int] = (3, 12, 0)) -> str:
    seen: set[str] = set()
    for candidate in [preferred, *_candidate_python_paths()]:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        exe = shutil.which(candidate) or str(candidate)
        try:
            p = subprocess.run(
                [exe, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if p.returncode != 0 or not p.stdout.strip():
                continue
            ver = _parse_py_version(p.stdout.strip())
            if ver >= min_version:
                return exe
        except Exception:
            continue
    raise SystemExit("No suitable python executable found. Install Python 3.12+ or set --python / env SKNOW_PYTHON.")


def _python_for_run(preferred: str | None = None) -> str:
    return os.environ.get("SKNOW_PYTHON", _find_python(preferred=preferred))

def _canonical_url(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


def _signature(value: str | None) -> str:
    if not value:
        return ""
    return sha256(value.encode("utf-8")).hexdigest()[:16]


def _truncate(value: str, limit: int = 1600) -> str:
    if value is None:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _build_logger(log_root: Path, run_tag: str, level: str = "INFO", log_file: str | None = None) -> tuple[logging.Logger, Path]:
    log_root.mkdir(parents=True, exist_ok=True)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = log_root / f"sknow_{run_tag}.log"

    logger = logging.getLogger(f"sknow.{run_tag}")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger, log_path


def _event_log_path(log_root: Path) -> Path:
    return log_root / "events.jsonl"


def _write_event(log_root: Path, stage: str, **payload) -> None:
    event = {"ts": datetime.now(timezone.utc).astimezone().isoformat(), "stage": stage, **payload}
    event_path = _event_log_path(log_root)
    with event_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")



def _safe_read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except Exception:
                continue
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="clue_news_collector wide collect -> clue_letter_dev2 shared total pool -> user rendering")

    # Backward-compatible user entrypoint.
    p.add_argument("--user-code", default=None, help="legacy: target only this user; if omitted, process all active users")

    # Collector config
    p.add_argument("--collector-script", default=None, help="override collector script path (ex: run_wide_collect.py)")
    p.add_argument("--manifest", default="config/wide_collect_manifest.v1.json", help="collector manifest path")
    p.add_argument("--run-prefix", default=None, help="collector run-prefix for artifact names")
    p.add_argument("--issue", default=None, help="YYYY-MM-DD used for shared total pool filename")
    p.add_argument("--issue-number", type=int, default=None, help="legacy passthrough for compatibility")
    p.add_argument("--top-items", type=int, default=20, help="legacy alias for collector per-query limits")

    p.add_argument("--skip-collect", action="store_true", help="skip collector; reuse existing shared total pool")
    p.add_argument("--collect", action="store_true", help="force collect (default when neither skip nor collect set)")
    p.add_argument("--collect-override", action="store_true", help="alias for force collect")
    p.add_argument("--no-overwrite-candidates", dest="overwrite_candidates", action="store_false", help="do not clean raw candidate artifacts before collect")

    # Paths / env
    p.add_argument("--report-dir", default=None, help="collector validation report directory")
    p.add_argument("--dev2-root", default=None, help="clue_letter_dev2 root path")
    p.add_argument("--dev2-collect-script", default=None, help="override delivery script path (service.py)")

    # Execution policy
    p.add_argument("--parallel-workers", type=int, default=3, help="parallel workers for user writing")
    p.add_argument("--batch-size", type=int, default=0, help="if >0, run only first N users from active user set")
    p.add_argument("--batch-start", type=int, default=0, help="zero-based start index for user batch")
    p.add_argument("--send", action="store_true", default=False, help="run final delivery after rendering")
    p.add_argument("--delivery-mode", default="gog", choices=["gog", "openclaw", "openclaw-only", "fallback", "local", "strict", "strict-gog"], help="delivery mode (gog/openclaw requires gog delivery; fallback keeps local file mode)")
    p.add_argument("--no-send", action="store_true", help="skip delivery")
    p.add_argument("--browser", action="store_true", help="use browser relay for dev2 writing stage")
    p.add_argument("--no-browser", action="store_true", help="force no browser for dev2 writing stage")
    p.add_argument("--collect-only", action="store_true", help="stop after shared pool creation")
    p.add_argument("--log-dir", default=None, help="directory for sknow run logs (default: <dev2-root>/logs/sknow)")
    p.add_argument("--log-file", default=None, help="override exact log file path")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="log level")
    p.add_argument("--python", default=None, help="python interpreter for collector/dev2 subcommands")

    # legacy output fields kept for compatibility
    p.add_argument("--to", default=None)
    p.add_argument("--account", default="dave170519@gmail.com")
    p.add_argument("--subject", default="[CLUE] AI BRIEFING | 오늘의 글로벌 핵심 동향")

    return p.parse_args()


def _clean_candidate_store(collector_root: Path) -> dict[str, int]:
    removed: dict[str, int] = {"raw_jsonl": 0, "curated_jsonl": 0, "known_urls": 0, "run_state": 0}

    targets = [
        collector_root / "data" / "ingest" / "raw",
        collector_root / "data" / "ingest" / "curated",
    ]
    for target in targets:
        if not target.exists():
            continue
        for p in target.glob("*.jsonl"):
            try:
                p.unlink()
                if "raw" in str(target):
                    removed["raw_jsonl"] += 1
                else:
                    removed["curated_jsonl"] += 1
            except Exception:
                pass

    # Known-URL ledger can cause next run to skip URLs even if raw is reset.
    # reset it when running full daily collect to make collector output a clean overwrite set.
    known_url_jsonl = collector_root / "data" / "known_urls" / "known_urls.jsonl"
    known_url_index = collector_root / "data" / "known_urls" / "known_urls_index.json"
    for p in (known_url_jsonl, known_url_index):
        if p.exists():
            try:
                p.unlink()
                removed["known_urls"] += 1
            except Exception:
                pass

    run_state = collector_root / "data" / "runs" / "current" / "run_state.json"
    if run_state.exists():
        try:
            run_state.unlink()
            removed["run_state"] += 1
        except Exception:
            pass

    return removed



def _find_collector_script(explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return path
        raise SystemExit(f"collector script not found: {path}")

    env_override = os.getenv("CLUE_NEWS_COLLECTOR_SCRIPT")
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.exists():
            return path
        raise SystemExit(f"collector script not found: {path}")

    base = Path(__file__).resolve().parents[2]
    candidate = base / "clue_news_collector" / "scripts" / "run_wide_collect.py"
    if candidate.exists():
        return candidate
    raise SystemExit("run_wide_collect.py not found. Set --collector-script explicitly")


def _run_collector(args: argparse.Namespace, env: dict[str, str], python_exec: str) -> tuple[Path, dict]:
    collector = _find_collector_script(args.collector_script)
    collector_root = collector.parent.parent
    manifest = (collector_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    report_dir = Path(args.report_dir) if args.report_dir else (collector_root / "logs" / "local_validation")
    report_dir = report_dir.resolve()
    run_prefix = args.run_prefix or f"runcollect2html_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        python_exec,
        str(collector),
        "--manifest",
        str(manifest),
        "--report-dir",
        str(report_dir),
        "--run-prefix",
        run_prefix,
        "--naver-display",
        str(args.top_items),
        "--brave-count",
        str(args.top_items),
        "--source-limit",
        "3",
    ]
    print(f"[sknow] run collector: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"collector failed\nstdout={proc.stdout}\nstderr={proc.stderr}")

    summary_path = report_dir / f"{run_prefix}.wide_collect.summary.json"
    if not summary_path.exists():
        # fallback: latest summary in report_dir (defensive)
        candidates = sorted(report_dir.glob("*.wide_collect.summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit(f"summary file not found: {summary_path}")
        summary_path = candidates[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary_path, summary


def _map_query(row: dict) -> str:
    ctx = row.get("query_context") if isinstance(row.get("query_context"), dict) else {}
    return (
        str(row.get("query") or "")
        or str(ctx.get("naver_query") or "")
        or str(ctx.get("google_query") or "")
        or str(ctx.get("brave_query") or "")
        or str(ctx.get("intent_cluster") or "")
        or str(row.get("lane") or "")
        or ""
    )


def _is_paid_candidate(text: str) -> bool:
    if not text:
        return False
    low = str(text).lower()
    paid_patterns = [
        r"\bpaywall\b",
        r"\bsubscription\b",
        r"\bsubscribe\b",
        r"\bsignin\b",
        r"\bsign\s*in\b",
        r"\blog\s*in\b",
        r"\b회원가입\b",
        r"\b유료\b",
        r"\b구독\b",
        r"\b멤버십\b",
        r"\b멤버\b",
        r"\bmember\b",
        r"\bpremium\b",
        r"\btrial\b",
        r"\b결제\b",
        r"\b유료\s*콘텐츠\b",
        r"\b구독\s*형\b",
    ]
    return any(re.search(pat, low, re.IGNORECASE) for pat in paid_patterns)


def _is_paid_url(url: str) -> bool:
    if not url:
        return False
    low = str(url).lower()
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()
    except Exception:
        return _is_paid_candidate(low)

    paid_domains = {
        "ft.com", "techcrunch.com", "seekingalpha.com", "wsj.com", "bloomberg.com", "reuters.com",
    }

    if any(t in path for t in ("/pay", "/pricing", "/payment", "/subscription", "/subscribe", "/membership", "/member", "/signin", "/login", "/signup", "/sign-up", "/paywall", "/register")):
        return True
    if any(t in query for t in ("paywall", "subscribe", "subscription", "member", "membership", "trial", "premium")):
        return True
    if any(d in host for d in paid_domains) and any(k in path for k in ("/member", "/subscription", "/subscribe", "/premium")):
        return True
    if _is_paid_candidate(query):
        return True
    return False


def _is_shared_row_filtered(row: dict) -> tuple[bool, str]:
    title = str(row.get("title_raw") or "")
    summary = str(row.get("text_summary") or row.get("summary") or "")
    body = str(row.get("fetch_text_raw") or row.get("body") or "")
    source = str(row.get("source") or "")
    lane = str(row.get("lane") or "")
    query = _map_query(row)
    url = _canonical_url(str(row.get("resolved_url") or row.get("discovered_url") or ""))

    texts = [title, summary, body, source, lane, query]
    joined = "\n".join(texts)

    # 운영 정책: digitimesasia 계열은 shared pool 적재 전 제외
    low_url = (url or "").lower()
    low_source = (source or "").lower()
    if "digitimesasia" in low_url or "digitimesasia" in low_source:
        return True, "blocked_source_digitimesasia"

    if _is_paid_candidate(joined):
        return True, "paid_keyword"
    if _is_paid_url(url):
        return True, "paid_url"
    if re.search(r"\bfree\s+trial\b|유료\s*뉴스|유료\s*구독", joined, re.IGNORECASE):
        return True, "paid_pattern"
    return False, ""



def _source_type(row: dict) -> str:
    raw_source = str(row.get("source") or "")
    if not raw_source:
        return "search"
    lowered = raw_source.lower()
    if "rss" in lowered or "feed" in lowered:
        return "rss"
    if "google" in lowered:
        return "google_news_rss"
    if "naver" in lowered:
        return "search"
    if "direct" in lowered:
        return "direct"
    return "search"


def _to_shared_rows(summary: dict, collector_root: Path) -> tuple[list[dict], dict[str, int]]:
    run_ids = []
    for item in summary.get("results", []):
        run_id = str(item.get("run_id") or "").strip()
        if run_id:
            run_ids.append(run_id)

    if not run_ids:
        return [], {"total": 0, "filtered": 0, "written": 0}

    seen_urls: set[str] = set()
    out: list[dict] = []
    filter_stats = {"total": 0, "filtered": 0, "written": 0}
    raw_dir = collector_root / "data" / "ingest" / "raw"

    for run_id in run_ids:
        for row in _safe_read_jsonl(raw_dir / f"{run_id}.jsonl"):
            url = _canonical_url(str(row.get("resolved_url") or row.get("discovered_url") or ""))
            if not url:
                continue
            filter_stats["total"] += 1
            is_filtered, reason = _is_shared_row_filtered(row)
            if is_filtered:
                filter_stats["filtered"] += 1
                continue
            sig = _signature(url + "|" + str(row.get("title_raw") or "") )
            if sig in seen_urls:
                continue
            seen_urls.add(sig)
            filter_stats["written"] += 1
            query = _map_query(row)
            ctx = row.get("query_context") if isinstance(row.get("query_context"), dict) else {}
            lane = str(row.get("lane") or ctx.get("intent_cluster") or "")
            body = str(row.get("fetch_text_raw") or "")
            summary_text = str(row.get("text_summary") or row.get("title_raw") or "")
            shared = {
                "record_id": row.get("record_id") or f"shared_{sig}",
                "run_id": row.get("run_id") or run_id,
                "url": url,
                "title": str(row.get("title_raw") or ""),
                "summary": summary_text[:900],
                "body": body,
                "body_snippet": body[:300],
                "published_at": str(row.get("published_at_raw") or ""),
                "source": str(row.get("source") or "OpenClaw Skill"),
                "source_type": _source_type(row),
                "lane": lane,
                "query": query,
                "need_category": lane,
                "country": "GLOBAL",
                "extraction_status": "success" if body else "failed",
                "origin_type": "clue_news_collector",
                "origin_detail": {
                    "manifest": str(ctx),
                    "run_id": str(run_id),
                    "collector_persona": str(row.get("persona") or ""),
                    "lane": lane,
                },
            }
            out.append(shared)

    return out, filter_stats


def _write_shared_pool(rows: list[dict], issue: str, dev2_root: Path) -> Path:
    shared_dir = dev2_root / "data" / "shared" / "daily news"
    shared_dir.mkdir(parents=True, exist_ok=True)
    out_path = shared_dir / f"total_news_{issue}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def _run_dev2(args: argparse.Namespace, env: dict[str, str], dev2_root: Path, action: str, user_code: str | None, send: bool, no_browser: bool, parallel: bool = False, workers: int = 2, python_exec: str = "python3") -> tuple[int, str, str, bool, datetime]:
    service = args.dev2_collect_script
    if service is None:
        service = str((dev2_root / "service.py").resolve())
    cmd: list[str] = [python_exec, service, action]
    if action == "run":
        if not user_code:
            raise ValueError("run action requires --user-code")
        cmd.extend(["--user-code", user_code])
    if not no_browser:
        # keep explicit browser-mode compatibility from sknow side; service defaults to HTTP requests.
        pass
    if args.browser:
        cmd.append("--browser")
    if args.no_browser:
        cmd.append("--no-browser")

    if send:
        cmd.append("--send")
    else:
        cmd.append("--no-send")

    if action == "run-all":
        if parallel and action == "run-all":
            cmd.append("--parallel")
            cmd.extend(["--workers", str(max(1, int(workers)))])

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(dev2_root), env=env)
    return proc.returncode, proc.stdout, proc.stderr, proc.returncode == 0, datetime.now(timezone.utc).astimezone()


def main() -> int:
    args = parse_args()
    env = _load_pipeline_env()

    if args.no_send and args.send:
        raise SystemExit("--send and --no-send cannot be used together")
    if not hasattr(args, "overwrite_candidates"):
        args.overwrite_candidates = True

    issue = args.issue or datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    run_tag = f"{issue}_{datetime.now(timezone.utc).astimezone().strftime('%H%M%S')}"
    default_log_root = Path(args.log_dir) if args.log_dir else (Path(args.dev2_root) if args.dev2_root else Path("/Users/davechoi/.openclaw/workspace/clue_letter_dev2")).resolve() / "logs" / "sknow"
    log_root = Path(default_log_root)

    # Configure logger & file logging
    logger, log_path = _build_logger(log_root, run_tag, level=args.log_level, log_file=args.log_file)
    logger.info("sknow started")
    logger.info("args=%s", json.dumps(vars(args), ensure_ascii=False))
    _write_event(log_root, "start", args=vars(args), issue=issue, run_tag=run_tag, python=env.get("PYTHON", None))

    _python = _python_for_run(args.python)
    # Enforce delivery policy for this pipeline run.
    # If actual send is requested, default to gog/openclaw-only.
    # For --no-send/dry runs, allow fallback to local file mode.
    delivery_mode = args.delivery_mode.lower()
    send_requested = bool(args.send and not args.no_send)
    if delivery_mode in {"gog", "openclaw", "openclaw-only", "strict", "strict-gog"}:
        env["SKNOW_DELIVERY_MODE"] = "openclaw-only" if send_requested else "fallback"
    elif delivery_mode in {"fallback", "local"}:
        env["SKNOW_DELIVERY_MODE"] = "fallback"
    else:
        env["SKNOW_DELIVERY_MODE"] = "openclaw-only" if send_requested else "fallback"
    env["PYTHON"] = _python
    dev2_root = Path(args.dev2_root) if args.dev2_root else Path("/Users/davechoi/.openclaw/workspace/clue_letter_dev2").resolve()
    _write_event(log_root, "resolved", dev2_root=str(dev2_root), python=_python)

    pipeline_start = time.perf_counter()
    if not args.skip_collect:
        collector_root = _find_collector_script(args.collector_script).parent.parent
        if args.overwrite_candidates:
            removed = _clean_candidate_store(collector_root)
            logger.info("candidate store cleaned: %s", removed)
            _write_event(log_root, "candidate_cleanup", **removed)
        else:
            removed = {"raw_jsonl": 0, "curated_jsonl": 0, "known_urls": 0, "run_state": 0}

        c0 = time.perf_counter()
        _summary_path, summary = _run_collector(args, env, _python)
        c1 = time.perf_counter()
        logger.info("collector complete: summary=%s, elapsed=%.2fs, path=%s", summary.get("run_prefix", "unknown"), c1 - c0, _summary_path)
        _write_event(
            log_root,
            "collect_complete",
            summary_path=str(_summary_path),
            elapsed_ms=int((c1 - c0) * 1000),
            results=len(summary.get("results", [])),
            raw_records=sum(int(row.get("raw_records_written", 0) or 0) for row in summary.get("results", [])),
            failures=len(summary.get("failures", [])),
            unique_urls=summary.get("unique_url_count", 0),
            run_prefix=summary.get("run_prefix"),
            removed=removed,
        )
        shared_rows, shared_filter_stats = _to_shared_rows(summary, collector_root)
        logger.info("shared rows transformed=%d filtered=%d", len(shared_rows), shared_filter_stats.get("filtered", 0))
        _write_event(log_root, "shared_filter", **shared_filter_stats)
        s0 = time.perf_counter()
        shared_pool = _write_shared_pool(shared_rows, issue, dev2_root)
        s1 = time.perf_counter()
        logger.info("shared total pool written=%s elapsed=%.2fs", shared_pool, s1 - s0)
        _write_event(
            log_root,
            "shared_pool_written",
            path=str(shared_pool),
            rows=len(shared_rows),
            issue=issue,
            elapsed_ms=int((s1 - s0) * 1000),
            run_records=len(summary.get("results", [])),
        )
        logger.info("total_candidates=%d run_records=%d", len(shared_rows), len(summary.get("results", [])))
    else:
        shared_pool = dev2_root / "data" / "shared" / "daily news" / f"total_news_{issue}.jsonl"
        if not shared_pool.exists():
            raise SystemExit(f"--skip-collect set but shared pool missing: {shared_pool}")
        logger.info("skip collect -> reuse shared pool: %s", shared_pool)
        _write_event(log_root, "skip_collect", shared_pool=str(shared_pool))

    if args.collect_only:
        result = {"ok": True, "mode": "collect-only", "issue": issue, "shared_pool": str(shared_pool)}
        logger.info("collect-only complete")
        _write_event(log_root, "collect_only", **result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    phase0 = time.perf_counter()
    if args.user_code:
        # legacy one-user path for compatibility
        stage = "run-user"
        code, out, err, _ok, t0 = _run_dev2(
            args,
            env,
            dev2_root,
            action="run",
            user_code=args.user_code,
            send=bool(args.send and not args.no_send),
            no_browser=args.no_browser,
            parallel=False,
            workers=1,
            python_exec=_python,
        )
        logger.info("%s phase finished code=%s elapsed=%.2fs", stage, code, (time.perf_counter() - phase0))
        _write_event(log_root, "dev2_phase", stage=stage, code=code, elapsed_ms=int((time.perf_counter() - phase0) * 1000), stdout=_truncate(out), stderr=_truncate(err))
    else:
        # shared-pool 기반 전체 사용자 rendering & delivery을 한 번의 run-all으로 통합.
        all_phase_start = time.perf_counter()
        code, out, err, _ok, _okt = _run_dev2(
            args,
            env,
            dev2_root,
            action="run-all",
            user_code=None,
            send=bool(args.send and not args.no_send),
            no_browser=args.no_browser,
            parallel=True,
            workers=max(1, args.parallel_workers),
            python_exec=_python,
        )
        stage_name = "dev2_run_all_send" if args.send and not args.no_send else "dev2_run_all_write"
        _write_event(
            log_root,
            stage_name,
            code=code,
            elapsed_ms=int((time.perf_counter() - all_phase_start) * 1000),
            stdout=_truncate(out),
            stderr=_truncate(err),
            user_code=None,
        )
        logger.info("run-all finished as single pass code=%s", code)
        if code != 0:
            raise SystemExit(f"dev2 run-all failed\nstdout={out}\nstderr={err}")

    total_elapsed = int((time.perf_counter() - pipeline_start) * 1000)
    if code != 0:
        raise SystemExit(f"dev2 phase failed\nstdout={out}\nstderr={err}")

    logger.info("sknow complete elapsed_ms=%d", total_elapsed)
    _write_event(log_root, "complete", elapsed_ms=total_elapsed, out=_truncate(out), shared_pool=str(shared_pool))
    print(out.strip())
    print(f"[sknow-log] {log_path}")
    print(f"[sknow-event-log] {log_root / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

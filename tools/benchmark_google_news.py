#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openclaw.agents.news_collector import NewsCollector
from clue_letter.agents.collection_agent import _normalize_title_signature, CollectionAgent
from clue_letter.agents.browser_adapter import HttpNewsAdapter


@dataclass
class BenchRow:
    user: str
    date: str
    raw_url: str
    clue_url: str
    open_url: str
    title: str


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        raw = str(row.get("record_link", "")).split("|", 1)[0].strip()
        if raw:
            row["record_url"] = raw
            rows.append(row)
    return rows


def _iter_user_days(root: Path, date: str) -> Iterable[tuple[str, Path]]:
    for user_dir in sorted(root.glob("*/")):
        if not user_dir.is_dir():
            continue
        daily_file = user_dir / "daily news" / f"{date}.jsonl"
        if daily_file.exists():
            yield (user_dir.name, daily_file)


def _dedupe_count(values: list[str]) -> tuple[int, int]:
    c = Counter([v for v in values if v])
    dup = sum(v - 1 for v in c.values() if v > 1)
    keys = sum(1 for v in c.values() if v > 1)
    return dup, keys


def _is_google_feed(url: str) -> bool:
    return "news.google.com/rss/articles" in (url or "").lower()


def _is_resolved_google(url: str) -> bool:
    u = (url or "").lower()
    return ("news.google.com" not in u) or ("/rss/articles" not in u)


def run(date: str, root: Path, with_open_resolve: bool = False) -> dict:
    collector = NewsCollector({})
    adapter = HttpNewsAdapter()
    ca = CollectionAgent(
        data_root=root.parent,
        needs_agent=type("N", (), {"get_user": lambda self, code: None, "list_users": lambda self: []}),
        fetch_body=lambda url: "",
        resolve_url=adapter.resolve_google_news_url,
    )

    rows_out: list[BenchRow] = []

    for user, daily in _iter_user_days(root, date):
        for row in _load_rows(daily):
            raw = row.get("record_url", "")
            title = (row.get("title") or "").strip()
            if not raw:
                continue

            clue_norm = ca._normalize_article_url(raw)
            open_norm = collector._canonicalize_url(raw)
            if with_open_resolve and _is_google_feed(raw):
                try:
                    open_norm = collector._resolve_source_url(raw)
                except Exception:
                    open_norm = open_norm

            rows_out.append(
                BenchRow(
                    user=user,
                    date=date,
                    raw_url=raw,
                    clue_url=clue_norm,
                    open_url=open_norm,
                    title=title,
                )
            )

    raw_urls = [r.raw_url for r in rows_out]
    clue_urls = [r.clue_url for r in rows_out]
    open_urls = [r.open_url for r in rows_out]
    title_keys = [_normalize_title_signature(r.title) for r in rows_out]

    raw_google = [u for u in raw_urls if _is_google_feed(u)]
    clue_resolved = [u for u in clue_urls if _is_resolved_google(u)]
    open_resolved = [u for u in open_urls if _is_resolved_google(u)]

    dup_raw, dup_raw_keys = _dedupe_count(raw_urls)
    dup_clue, dup_clue_keys = _dedupe_count(clue_urls)
    dup_open, dup_open_keys = _dedupe_count(open_urls)
    dup_title, dup_title_keys = _dedupe_count(title_keys)

    by_title_urls: dict[str, set[str]] = defaultdict(set)
    for r in rows_out:
        key = _normalize_title_signature(r.title)
        if not key:
            continue
        by_title_urls[key].add((r.clue_url or r.raw_url))

    same_title_multi_url = sum(1 for urls in by_title_urls.values() if len(urls) > 1)

    result = {
        "date": date,
        "total_rows": len(rows_out),
        "google_feed_count": len(raw_google),
        "raw_dup_count": dup_raw,
        "raw_dup_key_count": dup_raw_keys,
        "clue_dup_count": dup_clue,
        "clue_dup_key_count": dup_clue_keys,
        "open_dup_count": dup_open,
        "open_dup_key_count": dup_open_keys,
        "title_signature_dup_count": dup_title,
        "title_signature_dup_key_count": dup_title_keys,
        "title_signature_multi_url_count": same_title_multi_url,
        "clue_google_resolved_count": len(clue_resolved),
        "open_google_resolved_count": len(open_resolved),
        "sample_rows": [r.__dict__ for r in rows_out[:8]],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="벤치마크용 Google News 링크 정규화/중복 리포트")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--data-root", default="clue_letter/data", help="daily news 루트")
    parser.add_argument("--with-open-resolve", action="store_true", help="openclaw의 resolve 단계까지 수행")
    parser.add_argument("--out", default="", help="json 리포트 출력 경로")
    args = parser.parse_args()

    root = Path(args.data_root)
    rep = run(args.date, root, with_open_resolve=args.with_open_resolve)

    print(json.dumps(rep, ensure_ascii=False, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()

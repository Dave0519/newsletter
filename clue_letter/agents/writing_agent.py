from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import CollectedArticle, NewsletterEntry, UserProfile
from .utils import ensure_dir
from .llm_text_utils import practical_ko, extract_hashtags


def _load_template(template_path: Path) -> str:
    return Path(template_path).read_text(encoding="utf-8")


def _extract_block(text: str, start: str, end: str) -> str:
    a = text.find(start)
    b = text.find(end, a)
    if a == -1 or b == -1:
        raise ValueError(f"block {start}/{end} not found")
    return text[a + len(start) : b]


def _replace_block(text: str, start: str, end: str, replacement: str) -> str:
    a = text.find(start)
    b = text.find(end, a)
    if a == -1 or b == -1:
        raise ValueError(f"block {start}/{end} not found")
    return text[:a] + replacement + text[b + len(end) :]


def _safe_esc(s: str) -> str:
    return (s or "").replace("{{", "{").replace("}}", "}")


class WritingAgent:
    """수집 데이터 -> 공식 템플릿 변환 에이전트."""

    def __init__(self, template_path: Path, log_dir: Path):
        self.template_path = Path(template_path)
        self.log_dir = Path(log_dir)

    def _build_summary_points(self, entries: list[NewsletterEntry]) -> list[str]:
        # Summary 카테고리 요약: 토픽/제목 기반 5개 내외 리스트
        buckets: dict[str, list[str]] = {}
        for e in entries:
            topic = e.topic or "기타"
            buckets.setdefault(topic, []).append(e.title)

        out = []
        for topic, titles in buckets.items():
            out.append(f"• {topic}: " + ", ".join(titles[:2]))
        return out[:6]

    def _grouped_country_blocks(self, articles: list[NewsletterEntry]):
        countries: dict[str, list[NewsletterEntry]] = {}
        for a in articles:
            countries.setdefault(a.country or "GLOBAL", []).append(a)
        return countries

    def _derive_entry(self, c: CollectedArticle) -> NewsletterEntry:
        practical = practical_ko(c.title, c.summary)
        if not practical:
            practical = c.summary
        # 3~5문장 보정
        lines = [x.strip() for x in practical.replace("\n", " ").split(".") if x.strip()]
        practical = ". ".join(lines[:5]).strip()
        if practical and not practical.endswith("."):
            practical += "."

        return NewsletterEntry(
            title=c.title,
            summary=c.summary,
            url=c.url,
            country=c.country,
            practical_implication=practical,
            need_category=c.need_category,
            topic=c.need_category or "AI/반도체 동향",
        )

    def compose_html(self, user: UserProfile, collected: list[CollectedArticle], issue_number: int | None = None) -> str:
        if not collected:
            raise ValueError("No collected article")

        entries = [self._derive_entry(c) for c in collected]

        template = _load_template(self.template_path)
        template = template.replace("{{ISSUE_DATE}}", datetime.now().strftime("%Y. %m. %d"))
        template = template.replace("{{ISSUE_NUMBER}}", str(issue_number or 0).zfill(3))
        template = template.replace("{{SERIAL_NUMBER}}", user.user_code)

        hashtags = extract_hashtags([x for e in entries for x in (e.title, e.summary)], top_n=6)
        template = template.replace("{{NEEDS_HASHTAGS}}", hashtags)

        # Summary block (상단)
        summary_lines = "<br/>".join(self._build_summary_points(entries))
        summary_block = (
            '<p style="margin:0 0 10px 0;font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#FFFFFF;line-height:1.7">'
            f"요약: {summary_lines}</p>"
        )
        marker = "{{/ARTICLES}}"
        idx = template.find(marker)
        if idx != -1:
            template = template[:idx + len(marker)] + "\n" + summary_block + template[idx + len(marker):]

        # Country/article blocks
        countries = self._grouped_country_blocks(entries)
        row_t = _extract_block(template, "{{#COUNTRIES}}", "{{/COUNTRIES}}")
        art_t = _extract_block(row_t, "{{#ARTICLES}}", "{{/ARTICLES}}")
        rows = []
        for country, items in countries.items():
            row = row_t.replace("{{COUNTRY_NAME}}", country)
            ars = []
            for it in items:
                r = art_t
                r = r.replace("{{ARTICLE_TITLE}}", _safe_esc(it.title))
                r = r.replace("{{ARTICLE_SUMMARY}}", _safe_esc(it.summary))
                r = r.replace("{{ARTICLE_PRACTICAL_IMPLICATION}}", _safe_esc(it.practical_implication))
                r = r.replace("{{ARTICLE_LINK}}", it.url)
                ars.append(r)
            row = _replace_block(row, "{{#ARTICLES}}", "{{/ARTICLES}}", "\n".join(ars))
            rows.append(row)

        template = _replace_block(template, "{{#COUNTRIES}}", "{{/COUNTRIES}}", "\n".join(rows))
        return template

    def build_and_save(self, user: UserProfile, collected: list[CollectedArticle], issue_number: int | None = None, out_root: Path | None = None) -> Path:
        html = self.compose_html(user=user, collected=collected, issue_number=issue_number)

        out_root = Path(out_root or (self.log_dir / user.name.replace(" ", "_") / "outputs"))
        ensure_dir(out_root)
        date = datetime.now().strftime("%Y%m%d")
        out = out_root / f"{date}_{user.user_code}.html"
        out.write_text(html, encoding="utf-8")

        # issue no history
        (out_root / "issue_no.txt").write_text(str(int(issue_number or 0)), encoding="utf-8")

        # checksum + metadata
        digest = hashlib.sha1(html.encode("utf-8")).hexdigest()
        meta = {
            "user_code": user.user_code,
            "name": user.name,
            "issue_no": issue_number,
            "issue_date": datetime.now().strftime("%Y.%m.%d"),
            "count": len(collected),
            "sha1": digest,
        }
        (out_root / "meta.json").write_text(json_export(meta), encoding="utf-8")
        return out


def json_export(obj):
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)

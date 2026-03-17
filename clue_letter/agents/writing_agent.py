from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Callable
import re

from .models import CollectedArticle, NewsletterEntry, UserProfile
from .utils import ensure_dir
from .llm_text_utils import practical_ko, extract_hashtags, summarize_ko, rewrite_title


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

    def __init__(self, template_path: Path, log_dir: Path, fetch_body: Callable[[str], str] | None = None, resolve_url: Callable[[str], str | None] | None = None):
        self.template_path = Path(template_path)
        self.log_dir = Path(log_dir)
        self.fetch_body = fetch_body
        self.resolve_url = resolve_url

    def _clean_title(self, title: str) -> str:
        t = (title or "").strip()
        if not t:
            return t

        # source suffix 제거: e.g. "title | Source" / "title - Source"
        for sep in [" | ", " - ", " – ", " — ", " : "]:
            if sep in t:
                left, right = t.rsplit(sep, 1)
                # source 후보에는 보통 회사명/매체명이 오므로 길고 짧은 경우만 제거
                if right and 2 <= len(right) <= 40:
                    t = left.strip()
                    break
        # 괄호/브라켓 뒤쪽 부가 정보 절단
        for pat in [r"\s*\(.*\)$", r"\s*\[.*\]$"]:
            t = re.sub(pat, "", t).strip()
        return t

    def _korean_country(self, country: str) -> str:
        c = (country or "").strip().upper()
        mapping = {
            "KR": "한국",
            "KOREA": "한국",
            "US": "미국",
            "USA": "미국",
            "GLOBAL": "글로벌",
            "CN": "중국",
            "JP": "일본",
            "TW": "대만",
            "EU": "유럽",
            "HK": "홍콩",
            "IN": "인도",
            "DE": "독일",
            "FR": "프랑스",
            "UK": "영국",
        }
        return mapping.get(c, c or "글로벌")

    def _resolve_url(self, url: str) -> str:
        raw = (url or "").strip()
        if not raw:
            return ""
        if self.resolve_url:
            try:
                resolved = self.resolve_url(raw)
                if resolved:
                    return resolved
            except Exception:
                pass
        return raw

    def _ensure_korean_summary_lines(self, summary: str, max_lines: int = 5) -> str:
        if not summary:
            return summary
        lines = [x.strip() for x in summary.split("\n") if x.strip()]
        if len(lines) >= 2:
            return "\n".join(lines[:max_lines])
        parts = [p.strip() for p in summary.split(". ") if p.strip()]
        out = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not p.endswith("."):
                p += "."
            out.append(p)
            if len(out) >= max_lines:
                break
        return "\n".join(out)

    def _extract_needs(self, entries: list[NewsletterEntry]) -> list[str]:
        counts: Counter[str] = Counter()
        for e in entries:
            if e.need_category:
                counts[e.need_category] += 1
        if not counts:
            return []
        return [n for n, _ in counts.most_common(5)]

    def _build_need_hashtags(self, articles: list[CollectedArticle], max_n: int = 5) -> str:
        counts: Counter[str] = Counter()
        for a in articles:
            if a.need_category:
                counts[a.need_category] += 1
        if not counts:
            texts = [x for a in articles for x in (a.title, a.summary) if x]
            return extract_hashtags(texts, top_n=max_n)

        top_needs = [n for n, _ in counts.most_common(max_n)]
        return " ".join([f"#{x}" for x in top_needs])

    def _load_remote_body(self, url: str) -> str:
        if not self.fetch_body or not url:
            return ""
        try:
            text = self.fetch_body(url)
            return text or ""
        except Exception:
            return ""

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
        resolved_url = self._resolve_url(c.url)
        body = self._load_remote_body(resolved_url)
        raw_title = self._clean_title(c.title)

        rewritten_title = rewrite_title(raw_title, body or c.summary, max_chars=75)
        title = rewritten_title if rewritten_title else raw_title

        # 원문 본문 기반 4~5줄 요약을 DESCRIPTION으로 사용
        description = summarize_ko(body or c.summary, title=title, sentence_count=5)
        if not description:
            description = c.summary or "해당 기사를 본문에서 핵심 내용을 추출하지 못해 요약 텍스트가 비어 있습니다."
        description = self._ensure_korean_summary_lines(description, max_lines=5)

        practical = practical_ko(title, description, max_sentences=5)
        if not practical:
            practical = description

        return NewsletterEntry(
            title=title,
            summary=description,
            url=resolved_url,
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

        # header tags: use top needs directly from collected list
        # if insufficient needs, fallback to text keyword extraction
        need_tags = self._build_need_hashtags(collected)
        template = template.replace("{{NEEDS_HASHTAGS}}", need_tags)

        hashtags = self._build_need_hashtags(collected, max_n=6)
        if not hashtags:
            hashtags = extract_hashtags([x for e in entries for x in (e.title, e.summary)], top_n=6)
        template = template.replace("{{NEEDS_HASHTAGS}}", hashtags)

        # summary section
        summary_lines = "<br/>".join(self._build_summary_points(entries))
        summary_block = (
            f'<p style="margin:0 0 10px 0;font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#FFFFFF;line-height:1.7">'
            + f"요약: {summary_lines}</p>"
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
            row = row_t.replace("{{COUNTRY_NAME}}", self._korean_country(country))
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

        # placeholders that may be introduced later
        template = template.replace("{{CORE_DESCRIPTION}}", "오늘의 핵심 이슈를 니즈 기준으로 정리했습니다.")
        template = template.replace("{{GLOBAL_SCAN_INTRO}}", "")
        return template

    def build_and_save(
        self,
        user: UserProfile,
        collected: list[CollectedArticle],
        issue_number: int | None = None,
        out_root: Path | None = None,
    ) -> Path:
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

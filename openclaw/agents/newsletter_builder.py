from __future__ import annotations

import re
import os
from pathlib import Path
import hashlib
import requests

COUNTRY_LABEL = {
    "KR": "한국",
    "US": "미국",
    "CN": "중국",
    "TW": "대만",
    "GLOBAL": "글로벌",
}


class NewsletterBuilder:
    def __init__(self, template_path: str, country_order: list[str]):
        self.template_path = template_path
        self.country_order = country_order

    def template_fingerprint(self) -> tuple[str, str]:
        p = Path(self.template_path)
        raw = p.read_bytes()
        return (str(p.resolve()), hashlib.sha256(raw).hexdigest())

    def validate(self, scan: dict[str, list[dict]], research: list[dict], min_scan: int = 10, min_research: int = 2):
        total_scan = sum(len(v) for v in scan.values())
        errors = []
        if total_scan < min_scan:
            errors.append(f"global_scan<{min_scan}")
        if len(research) < min_research:
            errors.append(f"research<{min_research}")
        return (len(errors) == 0, errors)

    def build(
        self,
        scan: dict[str, list[dict]],
        research: list[dict],
        issue_date: str,
        issue_number: str = "001",
        serial_number: str = "",
        needs_hashtags: str = "",
        brand_mark: str = "SK hynix",
        global_scan_intro: str = "국가별 핵심 이슈를 선별했습니다.",
    ) -> str:
        html = Path(self.template_path).read_text(encoding="utf-8")

        country_block = self._extract_block(html, "{{#COUNTRIES}}", "{{/COUNTRIES}}")
        rendered_countries = []
        for c in self.country_order:
            items = scan.get(c, [])
            if not items:
                continue
            row = country_block.replace("{{COUNTRY_NAME}}", COUNTRY_LABEL.get(c, c))
            article_block = self._extract_block(row, "{{#ARTICLES}}", "{{/ARTICLES}}")
            article_rows = []
            for a in items:
                r = article_block
                summary_text = a.get("description", a.get("summary", ""))
                practical = a.get("practical_implication") or "관련 기술의 적용 범위, 비용 영향, 운영 리스크를 함께 점검할 필요가 있습니다."
                r = r.replace("{{ARTICLE_TITLE}}", self._esc(a.get("title_ko") or a.get("title", "")))
                r = r.replace("{{ARTICLE_SUMMARY}}", self._esc(summary_text))
                r = r.replace("{{ARTICLE_PRACTICAL_IMPLICATION}}", self._esc(practical))
                r = r.replace("{{ARTICLE_LINK}}", a.get("url", "#"))
                article_rows.append(r)
            row = self._replace_block(row, "{{#ARTICLES}}", "{{/ARTICLES}}", "\n".join(article_rows))
            rendered_countries.append(row)
        html = self._replace_block(html, "{{#COUNTRIES}}", "{{/COUNTRIES}}", "\n".join(rendered_countries))

        research_block = self._extract_block(html, "{{#RESEARCH_ITEMS}}", "{{/RESEARCH_ITEMS}}")
        research_rows = []
        for r in research:
            rr = research_block
            rr = rr.replace("{{RESEARCH_SOURCE}}", self._esc(r.get("source", "Unknown")))
            rr = rr.replace("{{RESEARCH_TITLE}}", self._esc(r.get("title", "")))
            rr = rr.replace("{{RESEARCH_SUMMARY}}", self._esc(r.get("description", r.get("summary", ""))))
            rr = rr.replace("{{RESEARCH_LINK}}", r.get("url", "#"))
            research_rows.append(rr)
        html = self._replace_block(html, "{{#RESEARCH_ITEMS}}", "{{/RESEARCH_ITEMS}}", "\n".join(research_rows))

        if not research_rows:
            html = re.sub(r"<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\s*SECTION 03[\s\S]*?<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\s*FOOTER", "<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n FOOTER", html)

        summary_bullets = self._build_summary(scan)
        summary_lines = [f"• {self._esc(x)}" for x in summary_bullets]
        summary_text = "<br><br>".join(summary_lines[:5]) if summary_lines else "오늘은 본문 추출 가능한 주요 기사가 부족했습니다."

        replacements = {
            "{{ISSUE_DATE}}": issue_date,
            "{{ISSUE_NUMBER}}": issue_number,
            "{{SERIAL_NUMBER}}": serial_number,
            "{{NEEDS_HASHTAGS}}": needs_hashtags,
            "{{BRAND_MARK}}": brand_mark,
            "{{GLOBAL_SCAN_INTRO}}": global_scan_intro,
            "{{CORE_DESCRIPTION}}": summary_text,
        }
        for k, v in replacements.items():
            html = html.replace(k, v)

        html = re.sub(r"CLUE 데일리 브리핑\s*·\s*[^·<]+\s*·", f"CLUE 데일리 브리핑 · {issue_date} ·", html)
        return html

    def _build_summary(self, scan: dict[str, list[dict]]) -> list[str]:
        bullets = []
        for c in self.country_order:
            for a in scan.get(c, []):
                d = " ".join((a.get("description") or "").split())
                if not d:
                    continue
                parts = [p.strip() for p in re.split(r"(?<=[.!?다])\s+", d) if p.strip()]
                if len(parts) >= 2:
                    bullets.append(f"{parts[0]} {parts[1]}")
                else:
                    bullets.append(parts[0])
                if len(bullets) >= 5:
                    return bullets
        return bullets

    @staticmethod
    def _esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _extract_block(text: str, start: str, end: str) -> str:
        i = text.find(start)
        j = text.find(end)
        if i == -1 or j == -1 or j < i:
            return ""
        return text[i + len(start):j]

    @staticmethod
    def _replace_block(text: str, start: str, end: str, replacement: str) -> str:
        i = text.find(start)
        j = text.find(end)
        if i == -1 or j == -1 or j < i:
            return text
        return text[:i] + replacement + text[j + len(end):]

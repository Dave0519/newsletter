from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Callable
from urllib.parse import parse_qs, urlparse, urlencode
import re

from .models import CollectedArticle, NewsletterEntry, UserProfile
from .utils import ensure_dir
from .llm_text_utils import practical_ko, extract_hashtags, summarize_ko, rewrite_title, translate_ko


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
        return self._resolve_source_url(url)

    def _canonicalize_url(self, url: str) -> str:
        try:
            p = urlparse((url or "").strip())
            if not p.scheme or not p.netloc:
                return (url or "").strip()
            drop_prefix = ("utm_", "fbclid", "gclid", "oc", "ref", "ref_src")
            q = parse_qs(p.query, keep_blank_values=False)
            kept = []
            for k, vals in q.items():
                lk = (k or "").strip().lower()
                if lk in drop_prefix or any(lk.startswith(x) for x in drop_prefix):
                    continue
                for v in vals:
                    kept.append((k, v))
            query = urlencode(kept, doseq=True)
            clean = p._replace(query=query, fragment="")
            return clean.geturl().rstrip("?")
        except Exception:
            return (url or "").strip()

    def _resolve_source_url(self, url: str) -> str:
        raw = (url or "").strip()
        if not raw:
            return ""
        if "news.google.com" not in raw:
            return self._canonicalize_url(raw)

        try:
            parsed = urlparse(raw)
            qs = parse_qs(parsed.query)
            if qs.get("url") and qs["url"][0]:
                return self._canonicalize_url(qs["url"][0])
        except Exception:
            pass

        candidate = raw
        if self.resolve_url:
            try:
                candidate = self.resolve_url(raw) or raw
            except Exception:
                candidate = raw

        return self._canonicalize_url(candidate)

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

    def _force_korean(self, text: str, fallback_on_fail: str | None = None) -> str:
        src = (text or "").strip()
        if not src:
            return fallback_on_fail or ""
        if self._contains_korean(src):
            return src

        translated = translate_ko(src)
        if translated and self._contains_korean(translated) and translated.strip() != src.strip():
            return translated

        translated = translate_ko(f"아래 영문 문장을 한국어로 자연스럽게 번역해줘: {src}")
        if translated and self._contains_korean(translated) and "영어" not in translated and "아래 영문" not in translated:
            return translated

        return fallback_on_fail or src

    def _contains_korean(self, text: str) -> bool:
        return bool(re.search(r"[가-힣]", text or ""))

    def _is_google_wrapper_url(self, url: str) -> bool:
        u = (url or "").lower()
        return "news.google.com/rss/articles" in u or "news.google.com" in u

    def _strip_html(self, text: str) -> str:
        if not text:
            return ""
        t = re.sub(r"<[^>]+>", " ", text)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _looks_google_noise(self, text: str) -> bool:
        t = (text or "").lower()
        if "google news" in t and len(t) < 200:
            return True
        if any(x in t for x in ["google", "cookie", "로그인", "개인정보", "sign in", "menu"]):
            return len(t) < 160
        return False

    def _resolve_summary_source(self, c: CollectedArticle, body: str) -> str:
        if body and self._is_google_wrapper_url(c.url) and self._looks_google_noise(body[:2000]):
            body = ""
        if body and self._looks_google_noise(body[:2000]):
            body = ""
        if body:
            return body[:30000]

        fallback = self._strip_html(c.summary or "")
        if fallback and not self._looks_google_noise(fallback):
            return fallback
        return self._strip_html(c.title or "")

    def _normalize_title_candidate(self, text: str) -> str:
        s = (text or "").strip()
        if not s:
            return s
        # remove source suffixes and common wrappers
        for sep in [" | ", " - ", " – ", " — ", " : "]:
            if sep in s and any(mark in s for mark in ["|", "-", "–", "—", ":"]):
                parts = s.split(sep)
                if len(parts) > 1:
                    s = parts[0].strip()
        return s

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
        """Global scan 핵심 5개를 제목+요약 기반으로 정렬해 반환."""
        out = []
        for idx, e in enumerate(entries[:5], 1):
            title = (e.title or "").strip()
            sum_txt = (e.summary or "").strip()
            if not sum_txt:
                sum_txt = title
            row = f"{idx}. {title}"
            if sum_txt:
                # 너무 길면 축약
                short = sum_txt.replace("\n", " ").strip()
                if len(short) > 150:
                    short = short[:147].rstrip() + "..."
                row = f"{row} — {short}"
            out.append(row)
        return out

    def _grouped_country_blocks(self, articles: list[NewsletterEntry]):
        countries: dict[str, list[NewsletterEntry]] = {}
        for a in articles:
            countries.setdefault(a.country or "GLOBAL", []).append(a)
        return countries

    def _derive_entry(self, c: CollectedArticle) -> NewsletterEntry:
        resolved_url = self._resolve_url(c.url)
        body = self._load_remote_body(resolved_url)
        if not body and resolved_url != c.url:
            body = self._load_remote_body(c.url)
        raw_title = self._normalize_title_candidate(self._clean_title(c.title))
        summary_source = self._resolve_summary_source(c, body)

        rewritten_title = rewrite_title(raw_title, summary_source, max_chars=75)
        if not rewritten_title:
            rewritten_title = translate_ko(raw_title)

        # 최종 제목은 반드시 한국어로 보정
        rewritten_title = self._force_korean(rewritten_title or raw_title, fallback_on_fail="제목 변환이 아직 완료되지 않았습니다.")
        title = rewritten_title if rewritten_title and self._contains_korean(rewritten_title) else self._force_korean(raw_title, fallback_on_fail="제목 변환이 아직 완료되지 않았습니다.")

        # 원문 본문 기반 4~5줄 요약을 DESCRIPTION으로 사용 (Google 래퍼면 summary 대체)
        description = summarize_ko(summary_source, title=title, sentence_count=5)
        if not description:
            description = self._strip_html(summary_source)

        description = self._force_korean(description, fallback_on_fail="")
        if not description:
            description = self._force_korean(self._strip_html(c.summary or raw_title), fallback_on_fail="")
        if not description:
            description = "해당 기사를 본문에서 핵심 내용을 추출하지 못해 요약 텍스트가 비어 있습니다."

        # 원문 근거 기반 4~5줄 내외 요약, 줄바꿈 정규화
        description = self._ensure_korean_summary_lines(description, max_lines=5)
        description = self._force_korean(description, fallback_on_fail="해당 기사를 본문에서 핵심 내용을 추출하지 못해 요약 텍스트가 비어 있습니다.")

        practical = practical_ko(title, description, max_sentences=5)
        practical = self._force_korean(practical, fallback_on_fail=description)
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
        template = template.replace("{{BRAND_MARK}}", "SK HYNIX")

        # header tags: use top needs directly from collected list
        # if insufficient needs, fallback to text keyword extraction
        need_tags = self._build_need_hashtags(collected)
        template = template.replace("{{NEEDS_HASHTAGS}}", need_tags)

        hashtags = self._build_need_hashtags(collected, max_n=6)
        if not hashtags:
            hashtags = extract_hashtags([x for e in entries for x in (e.title, e.summary)], top_n=6)
        template = template.replace("{{NEEDS_HASHTAGS}}", hashtags)

        # summary section (global scan 핵심 5개 기반)
        summary_lines = self._build_summary_points(entries)
        summary_block = "요약: " + "<br/>".join(summary_lines)
        template = template.replace("{{CORE_DESCRIPTION}}", summary_block)

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

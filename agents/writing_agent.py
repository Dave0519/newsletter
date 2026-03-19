from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Callable
from urllib.parse import parse_qs, urlparse, urlencode
import re
import html as _html

import requests

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

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

    def _normalize_practical_lines(self, text: str, max_lines: int = 5) -> str:
        if not text:
            return text
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[0-9]+[\)\.]|[가-힣]+\)|[①-⑳]|[#•\-*]\s*|첫째|둘째|셋째|넷째|다섯째|여섯째)\s*[:\-]?\s*", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
        return "\n".join(lines)

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

    def _clean_text(self, text: str) -> str:
        return self._strip_html(text)

    def _looks_google_noise(self, text: str) -> bool:
        t = (text or "").lower()
        if "google news" in t and len(t) < 200:
            return True
        if any(x in t for x in ["google", "cookie", "로그인", "개인정보", "sign in", "menu"]):
            return len(t) < 160
        return False


    def _llm_request(self, prompt: str, max_tokens: int = 500, temperature: float = 0.2) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return ""

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if str(model).lower().startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if r.status_code != 200:
                return ""
            return ((r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip())
        except Exception:
            return ""

    def _extract_readable_text(self, html_text: str) -> str:
        if not html_text:
            return ""

        if BeautifulSoup is None:
            return self._strip_html(html_text)

        try:
            soup = BeautifulSoup(html_text, "html.parser")

            # meta 기반 본문 단서 (설명/요약)
            for meta_name in ["description", "og:description", "twitter:description"]:
                m = soup.find("meta", attrs={"name": meta_name}) or soup.find("meta", property=meta_name)
                if m and m.get("content"):
                    meta_txt = self._clean_text(m.get("content") or "")
                    if meta_txt and len(meta_txt) >= 60:
                        return meta_txt

            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "iframe", "svg", "canvas"]):
                tag.decompose()

            noise_words = ["로그인", "회원가입", "로그인/회원가입", "기사검색", "전체뉴스", "기자검색", "메뉴", "카테고리", "전체서비스", "search", "javascript", "adsense", "광고", "쿠키", "personal"]
            candidates = []

            for node in soup.find_all(["article", "main", "section", "div"]):
                txts = []
                for p in node.find_all(["p", "li", "h1", "h2", "h3"]):
                    t = self._clean_text((p.get_text(" ", strip=True)))
                    if len(t) < 30:
                        continue
                    txts.append(t)

                if not txts:
                    continue
                txt = " ".join(txts)
                txt = re.sub(r"\s+", " ", txt).strip()
                if not txt or len(txt) < 240:
                    continue

                score = len(txt)
                low = txt.lower()
                if node.name in {"article", "main"}:
                    score += 160
                for w in noise_words:
                    score -= low.count(w) * 90

                candidates.append((score, txt))

            if not candidates:
                return self._strip_html(html_text)

            candidates.sort(key=lambda x: x[0], reverse=True)
            text_body = candidates[0][1]
            if len(text_body) < 220:
                return self._strip_html(html_text)
            return text_body
        except Exception:
            return self._strip_html(html_text)

    def _llm_generate_title_from_body(self, raw_title: str, body: str) -> str:
        base_title = self._clean_title(raw_title)
        source = (base_title or "").strip()
        if not body:
            return source

        prompt = (
            "너는 뉴스레터 에디터다. 기사 본문과 원문 제목 정보를 바탕으로 한국어 제목을 1줄로 작성해줘.\n"
            "출력은 제목 한 줄만.\n\n"
            f"[원문 제목]\n{source}\n\n"
            f"[기사 본문]\n{body[:5000]}"
        )
        out = self._llm_request(prompt, max_tokens=90, temperature=0.1)
        if not out:
            return source
        s = " ".join((out or "").strip().split())
        return s

    def _llm_summary_from_body(self, title: str, body: str, regenerate_hint: str = "", line_count: int = 6) -> str:
        if not body:
            return ""
        hint = f"\n[추가 교정 지시] {regenerate_hint}" if regenerate_hint else ""
        lc = max(5, int(line_count))
        prompt = (
            "너는 한국어 뉴스레터 에디터다. 기사 본문을 근거로 5~7줄 내외로 요약해줘.\n"
            "규칙:\n"
            "1) 본문 근거만 사용\n"
            "2) 추측/과장/메타문구 금지\n"
            "3) 핵심 사실-수치-영향 순으로 짧은 문장으로 정리\n"
            "4) 출력을 요약 본문만 출력\n"
            "5) 불필요한 소개/결론 문구 없이 핵심만 정리\n"
            "6) 본문에 등장한 숫자/단위(예: %, 조/억/만, $, €, $, 달러, USD, 원, 위안, 위안화, KB, %)는 절대 임의 변환하지 말고 본문과 동일한 숫자/단위로 작성\n"
            f"{hint}\n\n"
            f"[제목] {title}\n"
            f"[요약 줄 수] 대략 {lc}줄\n\n"
            f"[본문] {body[:12000]}"
        )
        return self._llm_request(prompt, max_tokens=950, temperature=0.2)

    def _extract_number_unit_tokens(self, text: str) -> set[str]:
        if not text:
            return set()

        pattern = re.compile(
            r"\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|□)?\b(?:\s*(?:조|억|만|천|백만|만개|개|명|개국|개사|억달러|조원|억원|만원|원|달러|달러\)|USD|US\$|\$|KRW|원화|위안|위안화|유로|EUR|엔|JPY))?"
        )
        # Fallback: capture percent and money patterns like 12.3%, $1.2조
        tokens = set()
        for m in pattern.findall(text or ""):
            t = re.sub(r"\s+", "", m)
            if t:
                tokens.add(t)
        return tokens

    def _normalize_num_token(self, token: str) -> str:
        return re.sub(r"[\s,]", "", token).replace("억원", "")

    def _llm_judge_summary(self, title: str, body: str, summary: str) -> tuple[bool, str]:
        if not title or not summary:
            return False, "empty_input"

        body_nums = self._extract_number_unit_tokens(body or "")
        summary_nums = self._extract_number_unit_tokens(summary or "")
        bad_nums: list[str] = []
        norm_body = {self._normalize_num_token(x): x for x in body_nums}
        for s in summary_nums:
            ns = self._normalize_num_token(s)
            if ns and ns not in norm_body:
                bad_nums.append(s)

        if bad_nums:
            return False, f"summary_has_unmatched_numbers:{','.join(sorted(set(bad_nums)))}"

        prompt = (
            "너는 요약 검수자다. 아래 요약이 본문 근거에 충실한지 판정해줘.\n"
            "조건: 1) 본문 근거 기반, 2) 한국어, 3) 과장/추측 없음, 4) 메타 문구 없음, 5) 본문에 등장한 숫자/단위는 요약에 동일 표기로만 사용\n"
            "출력은 JSON 한 줄: {\"pass\": true|false, \"reason\": \"...\"}\n\n"
            f"[제목]\n{title}\n\n"
            f"[본문]\n{body[:8000]}\n\n"
            f"[요약]\n{summary}"
        )
        raw = self._llm_request(prompt, max_tokens=180, temperature=0.0)
        if not raw:
            return True, "judge_skip"
        m_pass = re.search(r'"pass"\s*:\s*(true|false)', raw, re.I)
        m_reason = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
        passed = True
        if m_pass:
            passed = m_pass.group(1).lower() == "true"
        reason = m_reason.group(1) if m_reason else ""
        return passed, reason

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

    def _build_need_hashtags(
        self,
        articles: list[CollectedArticle | NewsletterEntry],
        user: UserProfile | None = None,
        max_n: int = 5,
    ) -> str:
        counts: Counter[str] = Counter()

        # 사용자 니즈 정의: id/need_text/aliases
        need_map_by_id: dict[str, dict[str, list[str] | str | None]] = {}
        if user and getattr(user, "needs_list", None):
            for item in user.needs_list:
                if not isinstance(item, dict):
                    continue
                nid = str(item.get("need_id", "")).strip()
                need_text = str(item.get("need_text", "")).strip()
                aliases = [str(a).strip() for a in item.get("aliases", []) if str(a).strip()]
                if nid:
                    need_map_by_id[nid] = {
                        "need_text": need_text,
                        "aliases": aliases,
                    }

        # 공통 stopword
        stopwords = {
            "있다", "있으며", "있어", "있을", "있습니다", "있을", "때", "등", "및", "하고", "한다", "할", "을", "를", "의", "가", "이", "에", "와", "과", "은", "는", "들", "이슈", "기사", "오늘", "기자", "공지",
            "nbsp", "nbsp;", "font", "style", "color", "width", "height", "class", "table", "tr", "td", "div", "span", "meta", "href", "src", "id", "body", "html", "head", "link", "script", "title", "http", "https", "amp", "news", "amp;", "업데이트", "브리핑", "요약", "정리", "리포트", "동향", "scan", "summary", "view", "report", "tech", "Tech", "개발", "솔루션", "top", "tier", "top-tier", "Top-tier",
            "in", "to", "on", "of", "for", "the", "a", "an", "and", "or", "with", "is", "it", "as", "at", "by", "from", "that", "this", "be", "are"
        }

        def add_tag(tag: str, weight: int = 1) -> None:
            t = (tag or "").strip()
            if not t:
                return
            t = _html.unescape(t)
            t = re.sub(r"&[A-Za-z]+;", "", t)
            t = t.replace("nbsp", "").replace("&", "").strip()
            t = re.sub(r"\s+", " ", t).strip()
            tl = t.lower()

            # --- R&D / 내부 분류 태그 정규화 ---
            tl_compact = re.sub(r"[^a-z0-9]", "", tl)

            # (1) 브랜드/조직명 기반 장문 태그는 제외 (예: SKHYNIXGLOBALRD)
            if "skhynix" in tl_compact or ("sk" in tl_compact and "hynix" in tl_compact):
                return

            # (2) R&D 변형 감지: r&d, r and d, rnd, researchdevelopment 등
            is_rd_variant = (
                tl_compact in {"rd", "rnd", "researchdevelopment"} or
                "researchdevelopment" in tl_compact
            )

            # (3) 내부 분류형(특히 영문+GLOBAL/PRODUCT+RD)은 제외
            ascii_only = bool(re.fullmatch(r"[A-Za-z0-9&\-\s]+", t))
            if ascii_only:
                # ProductRD / GlobalRD / ...GLOBALRD 류는 내부 분류로 보고 제외
                if (("product" in tl_compact) or ("global" in tl_compact)) and (tl_compact.endswith("rd") or "rnd" in tl_compact):
                    return
                # 대문자 장문 태그는 제외 (브랜딩/조직코드일 확률 큼)
                if t.isupper() and len(t) >= 10:
                    return

            # (4) R&D는 "살린다": 단, 뒤에 붙는 조직/역량/요소기술 같은 꼬리는 잘라서 깔끔하게 만든다.
            if is_rd_variant:
                # 표준 표기 선택: 원하면 "RnD"로 바꿔도 됨
                RD_CANON = "R&D"

                # 케이스 A: 순수 R&D라면 그대로
                if tl_compact in {"rd", "rnd", "researchdevelopment"} and len(t.strip()) <= 12:
                    t = RD_CANON
                    tl = t.lower()
                else:
                    # 케이스 B: 한글 prefix + R&D + (꼬리) 형태면 "prefix+R&D"까지만 살림
                    # 예: "메모리R&D핵심역량" -> "메모리R&D"
                    # 우선 흔한 변형들을 R&D로 치환
                    norm = re.sub(r"(?i)r\s*&\s*d", RD_CANON, t)
                    norm = re.sub(r"(?i)\br\s*and\s*d\b", RD_CANON, norm)
                    norm = re.sub(r"(?i)\brnd\b", RD_CANON, norm)

                    idx = norm.upper().find("R&D")
                    if idx != -1:
                        prefix = norm[:idx].strip()
                        # prefix가 너무 길거나 영문 위주면 내부 분류 가능성이 있어 제한
                        if prefix and len(prefix) > 16 and not re.search(r"[가-힣]", prefix):
                            return
                        t = f"{prefix}{RD_CANON}" if prefix else RD_CANON
                        tl = t.lower()
                    else:
                        # 못 찾으면 그냥 표준 R&D로 통일
                        t = RD_CANON

            if not t or len(t) < 2 or t in stopwords or tl in stopwords:
                return
            # URL/도메인/트래킹 성격 토큰 강제 제외
            if any(x in tl for x in ("http://", "https://", "www.", ".com", ".net", ".io", ".co", "utm_", "fbclid", "gclid")):
                return
            if any(x in t for x in ("/", "?", "=")):
                return

            # 공백 포함 태그 처리:
            # - "AI Agent" 같은 의미 있는 조합은 붙여쓰기("AIAgent")로 정규화
            # - 단, 범용 단어(tech/개발/솔루션/동향/업데이트 등)가 섞이면 노이즈로 보고 버림
            if " " in t:
                parts = [p for p in t.split(" ") if p]
                bad_parts = {"tech", "개발", "솔루션", "동향", "업데이트", "뉴스", "요약", "정리", "리포트", "top", "tier", "top-tier"}
                if any(p.lower() in bad_parts for p in parts):
                    return
                t = "".join(parts)
                tl = t.lower()

            if not re.fullmatch(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9\s#&+\-_/\.]*", t):
                return
            is_ko = bool(re.search(r"[가-힣]", t))
            if not is_ko:
                if re.fullmatch(r"[A-Za-z0-9]+", t):
                    if len(t) < 3 and t.upper() != "AI":
                        return
            counts[t] += weight

        # 1) 매칭 니즈 기반으로 태그 우선 반영
        for a in articles:
            # article-level 기본 카테고리/토픽
            if getattr(a, "need_category", None):
                add_tag(str(a.need_category).strip(), weight=3)
            if getattr(a, "topic", None):
                add_tag(str(a.topic).strip(), weight=1)

            for nid in getattr(a, "matched_need_ids", []) or []:
                item = need_map_by_id.get(str(nid).strip())
                if item and item.get("need_text"):
                    add_tag(str(item["need_text"]).strip(), weight=4)
                if item and item.get("aliases"):
                    for al in item["aliases"]:  # type: ignore[arg-type]
                        add_tag(str(al).strip(), weight=2)

            for alias in getattr(a, "matched_aliases", []) or []:
                add_tag(str(alias), weight=2)

            for need in getattr(a, "matched_needs", []) or []:
                add_tag(str(need), weight=3)

        # 2) 부족분 채우기: 기사 제목/요약 토큰에서 뽑지 말고, 사용자 니즈에서만 보충 (노이즈 방지)
        if len(counts) < 2:
            if user and getattr(user, "needs_list", None):
                for item in user.needs_list:
                    if not isinstance(item, dict):
                        continue
                    nt = str(item.get("need_text", "")).strip()
                    if nt:
                        add_tag(nt, weight=2)
            elif user and getattr(user, "interests", None):
                for it in user.interests:
                    it = str(it).strip()
                    if it:
                        add_tag(it, weight=2)

        if not counts:
            return ""

        ordered = [k for k, _ in counts.most_common()]
        unique_tags: list[str] = []
        for x in ordered:
            if x and x not in unique_tags:
                unique_tags.append(x)
            if len(unique_tags) >= max_n:
                break

        def _to_hashtag_token(s: str) -> str:
            s = (s or "").strip()
            s = s.replace("#", "")
            s = re.sub(r"\s+", "", s)  # 공백 제거(붙여쓰기)
            return s

        final: list[str] = []
        seen_norm: set[str] = set()
        for x in unique_tags:
            tok = _to_hashtag_token(x)
            if not tok:
                continue
            norm = tok.lower()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            final.append(tok)
            if len(final) >= max_n:
                break

        return " ".join([f"#{x}" for x in final])



    def _judge_country_by_title(self, title: str) -> str:
        """Use LLM to classify article title into one of KR/US/CN/TW/GLOBAL."""
        base = (title or "").strip()
        if not base:
            return "GLOBAL"

        prompt = (
            "아래 기사 제목만 보고 뉴스가 다뤄지는 주요 국가 맥락을 판단해라.\\n"
            "반드시 아래 5개 중 정확히 1개만 답해.\\n"
            "옵션: KR, US, CN, TW, GLOBAL\\n\\n"
            "규칙:\\n"
            "- 한국 관련이면 KR\\n"
            "- 미국 관련이면 US\\n"
            "- 중국 관련이면 CN\\n"
            "- 대만 관련이면 TW\\n"
            "- 위 4개에 명확히 안 걸리면 GLOBAL\\n\\n"
            f"[기사 제목]\\n{base}"
        )
        out = (self._llm_request(prompt, max_tokens=16, temperature=0.0) or "").strip().upper()
        if not out:
            return "GLOBAL"

        candidates = {"KR", "US", "CN", "TW", "GLOBAL"}
        for token in out.replace(",", " ").replace(";", " ").replace("/", " ").replace("\n", " ").split():
            t = token.strip().upper().strip(".\n ")
            if t in candidates:
                return t
        return "GLOBAL"

    def _coerce_country_code(self, country: str) -> str:
        c = (country or "").strip().upper()
        mapping = {
            "KOREA": "KR",
            "KOR": "KR",
            "KOREAN": "KR",
            "한국": "KR",
            "미국": "US",
            "USA": "US",
            "AMERICA": "US",
            "CHINA": "CN",
            "CHINESE": "CN",
            "중국": "CN",
            "TAIWAN": "TW",
            "TAIWAN": "TW",
            "대만": "TW",
            "글로벌": "GLOBAL",
            "GLOBAL": "GLOBAL",
        }
        return mapping.get(c, c or "GLOBAL") if c in mapping else c if c in {"KR", "US", "CN", "TW", "GLOBAL"} else "GLOBAL"


    def _load_remote_body(self, url: str) -> str:
        if not self.fetch_body or not url:
            return ""
        try:
            text = self.fetch_body(url)
            return text or ""
        except Exception:
            return ""

    def _build_summary_points(self, entries: list[NewsletterEntry]) -> str:
        """GLOBAL SCAN 렌더링 순서를 그대로 반영해 1~2줄 요약 리스트 구성."""
        lines: list[str] = []

        # GLOBAL SCAN에서 실제 표시되는 국가 블록/기사 순서를 유지
        grouped = self._grouped_country_blocks(entries)
        ordered_articles: list[NewsletterEntry] = []
        for _, arts in self._ordered_country_blocks(grouped).items():
            ordered_articles.extend(arts)

        for e in ordered_articles:
            title = (e.title or "").strip()
            summary = (e.summary or "").strip()

            compact = self._ensure_korean_summary_lines(summary, max_lines=2)
            compact = _safe_esc((compact or "").replace("\n", " ")).strip()
            if compact:
                lines.append(f"• {compact}")
                continue

            if title:
                lines.append(f"• {title}")

        return "<br/><br/>".join(lines) if lines else "오늘은 본문 추출 가능한 주요 기사가 부족했습니다."

    def _grouped_country_blocks(self, articles: list[NewsletterEntry]):
        countries: dict[str, list[NewsletterEntry]] = {}
        for a in articles:
            countries.setdefault(a.country or "GLOBAL", []).append(a)
        return countries

    def _ordered_country_blocks(self, countries: dict[str, list[NewsletterEntry]]):
        # global scan block order: 한국 -> 미국 -> 중국 -> 대만 -> 글로벌
        order = ["KR", "US", "CN", "TW", "GLOBAL"]
        # keep unknowns at end in discovery order
        out = OrderedDict()
        for c in order:
            if c in countries:
                out[c] = countries[c]
        for c, arts in countries.items():
            if c not in order and c not in out:
                out[c] = arts
        return out

    def _derive_entry_payload(self, c: CollectedArticle) -> tuple[NewsletterEntry, dict[str, str]]:
        resolved_url = self._resolve_url(c.url)
        body_html = self._load_remote_body(resolved_url)
        if not body_html and resolved_url != c.url:
            body_html = self._load_remote_body(c.url)
        readable_body = self._extract_readable_text(body_html) if body_html else ""

        raw_title = self._normalize_title_candidate(self._clean_title(c.title))
        summary_source = self._resolve_summary_source(c, readable_body)

        # 제목: 본문 기반 생성 우선
        rewritten_title = self._llm_generate_title_from_body(raw_title, summary_source)
        if not rewritten_title:
            rewritten_title = rewrite_title(raw_title, summary_source, max_chars=75)
        if not rewritten_title:
            rewritten_title = translate_ko(raw_title)

        rewritten_title = self._force_korean(rewritten_title or raw_title, fallback_on_fail="제목 변환이 아직 완료되지 않았습니다.")
        title = rewritten_title if rewritten_title and self._contains_korean(rewritten_title) else self._force_korean(raw_title, fallback_on_fail="제목 변환이 아직 완료되지 않았습니다.")

        # 요약: 본문 기반 5~7줄 + 검수 재시도 1회
        description = self._llm_summary_from_body(title, summary_source, line_count=7)
        if description:
            ok, reason = self._llm_judge_summary(title, summary_source, description)
            if not ok:
                description = self._llm_summary_from_body(title, summary_source, regenerate_hint=reason, line_count=7)

        if not description:
            description = summarize_ko(summary_source, title=title, sentence_count=5)
        if not description:
            description = self._strip_html(summary_source)

        description = self._force_korean(description, fallback_on_fail="")
        if not description:
            description = self._force_korean(self._strip_html(c.summary or raw_title), fallback_on_fail="")
        if not description:
            description = "해당 기사를 본문에서 핵심 내용을 추출하지 못해 요약 텍스트가 비어 있습니다."

        description = self._ensure_korean_summary_lines(description, max_lines=7)
        description = self._force_korean(description, fallback_on_fail="해당 기사를 본문에서 핵심 내용을 추출하지 못해 요약 텍스트가 비어 있습니다.")

        entry = NewsletterEntry(
            title=title,
            summary=description,
            url=resolved_url,
            country=c.country,
            practical_implication="",
            need_category=c.need_category,
            topic=c.need_category or "AI/반도체 동향",
        )
        ctx = {
            "summary_source": summary_source,
            "readable_body": readable_body,
            "raw_title": raw_title,
            "input_summary": c.summary or "",
        }
        return entry, ctx

    def _openclaw_style_practical(self, title: str, summary: str, body: str) -> str:
        if not title and not summary:
            return ""


        body_snippet = (body or "")[:8000]
        prompt = (
            "아래 기사 제목/요약/본문을 바탕으로 실무 시사점을 한국어로 작성해라.\n"
            "작성 원칙:\n"
            "1) 기사 본문 근거 중심으로만 작성, 외부 추측 최소화\n"
            "2) 해당 이슈에 대한 실무 액션/리스크/대응 방향을 3~5문장으로 제시\n"
            "3) 단정이 아닌 실행 관점 중심 문장으로 마무리\n"
            "4) 문장당 짧고 즉시 참고 가능한 수준\n"
            "5) 번호/리스트(1), 첫째, 둘째, • 등) 형식 사용 금지\n\n"
            f"[제목]\n{title}\n\n"
            f"[요약]\n{summary}\n\n"
            f"[본문]\n{body_snippet}"
        )
        out = self._llm_request(prompt, max_tokens=420, temperature=0.2)
        out = (out or "").strip()
        if not out:
            return ""
        return self._normalize_practical_lines(self._ensure_korean_summary_lines(out, max_lines=5), max_lines=5)



    def _attach_practical_implications(self, entries: list[NewsletterEntry], contexts: list[dict[str, str]]) -> list[NewsletterEntry]:
        out: list[NewsletterEntry] = []
        for e, ctx in zip(entries, contexts):
            practical_source = (ctx.get("summary_source") or "").strip()
            practical_body = (ctx.get("readable_body") or "").strip()

            practical = self._openclaw_style_practical(
                title=e.title,
                summary=e.summary,
                body=(practical_body or practical_source),
            )
            if not practical:
                practical = practical_ko(
                    e.title,
                    f"{e.summary}\n\n{practical_source}",
                    max_sentences=5,
                )
            e.practical_implication = practical or e.summary
            out.append(e)
        return out

    def _derive_entry(self, c: CollectedArticle) -> NewsletterEntry:
        entry, _ = self._derive_entry_payload(c)
        return entry


    def _build_trace_id(self, user: UserProfile, issue_number: int | None, article_count: int) -> str:
        issue = str(issue_number or 0).zfill(3)
        t = datetime.now().strftime("%Y%m%d%H%M%S")
        base = f"{t}|{user.user_code}|{issue}|{article_count}"
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
        return f"clue_{t}_{user.user_code}_{issue}_{digest}"

    def _inject_trace_comment(self, html: str, user: UserProfile, issue_number: int | None, article_count: int) -> str:
        trace_id = self._build_trace_id(user, issue_number, article_count)
        marker = (
            f"<!-- CLUE_INTERNAL_TRACE user_code={user.user_code} user={user.name} "
            f"issue={issue_number or 0} articles={article_count} trace_id={trace_id} -->"
        )
        if "</body>" in html:
            return html.replace("</body>", marker + "\n</body>")
        return html + marker

    def compose_html(self, user: UserProfile, collected: list[CollectedArticle], issue_number: int | None = None) -> str:
        if not collected:
            raise ValueError("No collected article")

        entries_payload = [self._derive_entry_payload(c) for c in collected]
        entries = [entry for entry, _ in entries_payload]
        entries = self._attach_practical_implications(entries, [ctx for _, ctx in entries_payload])

        template = _load_template(self.template_path)
        template = template.replace("{{ISSUE_DATE}}", datetime.now().strftime("%Y. %m. %d"))
        template = template.replace("{{ISSUE_NUMBER}}", str(issue_number or 0).zfill(3))
        template = template.replace("{{SERIAL_NUMBER}}", user.user_code)
        template = template.replace("{{BRAND_MARK}}", "SK HYNIX")

        # header tags: use top needs directly from collected list
        # if insufficient needs, fallback to text keyword extraction
        need_tags = self._build_need_hashtags(collected, user=user, max_n=5)
        template = template.replace("{{NEEDS_HASHTAGS}}", need_tags)

        # Country/article blocks
        for e in entries:
            judged = self._judge_country_by_title(e.title)
            e.country = self._coerce_country_code(judged or e.country)

        # summary section follows GLOBAL SCAN(국가 블록) 렌더 순서
        summary_lines = self._build_summary_points(entries)
        summary_block = summary_lines
        template = template.replace("{{CORE_DESCRIPTION}}", summary_block)

        countries = self._ordered_country_blocks(self._grouped_country_blocks(entries))
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
        return self._inject_trace_comment(template, user=user, issue_number=issue_number, article_count=len(entries))

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
            "trace_id": self._build_trace_id(user, issue_number, len(collected)),
        }
        (out_root / "meta.json").write_text(json_export(meta), encoding="utf-8")
        return out


def json_export(obj):
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)

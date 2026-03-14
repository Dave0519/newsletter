from __future__ import annotations

import os
import re
import json
from html import unescape
from typing import List, Dict, Tuple

import requests
from bs4 import BeautifulSoup


class ContentProcessor:
    """본문 기반 기사별 요약 생성기 (LLM 작성 + LLM Judge 검증)."""

    def __init__(self, llm_cfg: dict | None = None):
        self.cfg = llm_cfg or {}

    def process_news_batch(self, items: List[Dict], lang: str = "ko") -> List[Dict]:
        out: List[Dict] = []
        for it in items or []:
            rss_title = self._clean(it.get("title", ""))
            source_title, body, status = self._extract_article_content(it.get("url", ""))
            if status != "success" or len(body) < 400:
                continue

            # 제목은 반드시 원문 URL 페이지에서 추출한 값을 우선 사용
            title = self._clean(source_title or rss_title)

            summary = self._llm_summary_from_body(title=title, body=body)
            if not summary:
                continue

            # LLM-as-a-Judge: 본문 근거성/언어/환각 여부 확인
            judge_ok, judge_reason = self._llm_judge_summary(title=title, body=body, summary=summary)
            if not judge_ok:
                # 1회 재생성
                summary_retry = self._llm_summary_from_body(title=title, body=body, regenerate_hint=judge_reason)
                if not summary_retry:
                    continue
                judge_ok2, _ = self._llm_judge_summary(title=title, body=body, summary=summary_retry)
                if not judge_ok2:
                    continue
                summary = summary_retry

            # Stage D에서는 practical_implication을 생성하지 않는다.
            # 최종 기사 선정 이후(Stage F)에서만 생성한다.
            practical = ""

            row = dict(it)
            # 1:1:1 고정 필드
            row["title_from_url"] = title
            row["summary_from_body"] = summary
            row["source_url"] = it.get("url", "")
            row["article_body"] = body

            row["title_ko"] = self._llm_generate_title_from_body(url_title=title, body=body)
            row["description"] = summary
            row["practical_implication"] = practical
            row["extraction_status"] = "success"
            out.append(row)

        return out

    def process_research_batch(self, items: List[Dict], lang: str = "ko") -> List[Dict]:
        return items or []

    def generate_practical_implications(self, scan_by_country: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Generate practical implications after final article selection stage."""
        out: Dict[str, List[Dict]] = {}
        for c, items in (scan_by_country or {}).items():
            bucket: List[Dict] = []
            for it in items or []:
                row = dict(it)
                practical = (row.get("practical_implication") or "").strip()
                if not practical:
                    title = row.get("title_from_url") or row.get("title_ko") or row.get("title") or ""
                    summary = row.get("summary_from_body") or row.get("description") or ""
                    body = row.get("article_body") or ""
                    practical = self._llm_practical_implication(title=title, body=body, summary=summary)
                    if not practical:
                        practical = "해당 이슈는 관련 기술/공급망/투자 우선순위를 점검할 때 참고할 만한 신호입니다."
                    row["practical_implication"] = practical
                bucket.append(row)
            out[c] = bucket
        return out

    def _extract_article_content(self, url: str) -> Tuple[str, str, str]:
        if not url:
            return "", "", "fail"
        try:
            r = requests.get(url, timeout=20, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                return "", "", "fail"
            soup = BeautifulSoup(r.text, "html.parser")

            source_title = ""
            og = soup.find("meta", property="og:title")
            tw = soup.find("meta", attrs={"name": "twitter:title"})
            h1 = soup.find("h1")
            if og and og.get("content"):
                source_title = self._clean(og.get("content", ""))
            elif tw and tw.get("content"):
                source_title = self._clean(tw.get("content", ""))
            elif h1:
                source_title = self._clean(h1.get_text(" ", strip=True))
            elif soup.title:
                source_title = self._clean(soup.title.get_text(" ", strip=True))

            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
                tag.decompose()

            # article/main 우선
            candidates = []
            for node in soup.find_all(["article", "main", "section", "div"]):
                txts = []
                for p in node.find_all(["p", "li"]):
                    t = self._clean(p.get_text(" ", strip=True))
                    if len(t) >= 40:
                        txts.append(t)
                joined = "\n".join(txts)
                if len(joined) >= 300:
                    score = len(joined)
                    if node.name == "article":
                        score += 500
                    candidates.append((score, joined))

            if not candidates:
                return source_title, "", "fail"

            candidates.sort(key=lambda x: x[0], reverse=True)
            body = self._clean(candidates[0][1])
            return source_title, body, "success"
        except Exception:
            return "", "", "fail"

    def _llm_summary_from_body(self, title: str, body: str, regenerate_hint: str = "") -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return ""

        hint = f"\n추가 교정 지시: {regenerate_hint}" if regenerate_hint else ""
        prompt = (
            "너는 한국어 뉴스레터 에디터다. 반드시 '기사 본문'만 근거로 기사별 요약을 작성하라.\n"
            "규칙:\n"
            "1) 한국어로 작성\n"
            "2) 문장 길이/개수 제한 없음\n"
            "3) 추측/과장/기사에 없는 주장 금지\n"
            "4) 메타 표현 금지(예: 기사 본문에는, 이 기사는)\n"
            "5) 제목 반복이 아니라 본문 핵심 변화와 맥락을 압축\n"
            "6) 결과는 요약 본문만 출력\n"
            f"{hint}\n\n"
            f"[원제목]\n{title}\n\n"
            f"[기사 본문]\n{body[:12000]}"
        )
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 900,
                },
                timeout=40,
            )
            if r.status_code != 200:
                return ""
            txt = (r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            return self._clean(txt)
        except Exception:
            return ""

    def _llm_judge_summary(self, title: str, body: str, summary: str) -> Tuple[bool, str]:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return True, "no_api_key_skip"

        prompt = (
            "너는 요약 검수자다. 아래 요약이 기사 본문 근거에 충실한지 심사하라.\n"
            "판정 기준:\n"
            "- 본문 근거 기반(환각 없음)\n"
            "- 한국어 자연스러움\n"
            "- 과장/추측/메타표현 금지 준수\n"
            "출력 형식(JSON 한 줄): {\"pass\":true|false,\"reason\":\"...\"}\n\n"
            f"[제목]\n{title}\n\n"
            f"[본문]\n{body[:9000]}\n\n"
            f"[요약]\n{summary}"
        )
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 180,
                },
                timeout=30,
            )
            if r.status_code != 200:
                return True, "judge_api_fail_skip"
            raw = (r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            m_pass = re.search(r'"pass"\s*:\s*(true|false)', raw, re.I)
            m_reason = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
            passed = (m_pass.group(1).lower() == "true") if m_pass else True
            reason = m_reason.group(1) if m_reason else ""
            return passed, reason
        except Exception:
            return True, "judge_exception_skip"

    def _llm_judge_need_cluster_match(
        self,
        title: str,
        summary: str,
        body: str,
        cluster_name: str,
        terms: list[str],
    ) -> Tuple[bool, str, float]:
        """Return (matched, reason, score) for whether article satisfies a need cluster."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return False, "no_api_key_skip", 0.0

        norm_terms = [self._clean(t).strip() for t in (terms or []) if isinstance(t, str) and t.strip()]
        term_line = ", ".join(norm_terms[:12])

        prompt = (
            "너는 뉴스레터 품질 심사자다.\n"
            "아래 '클러스터 니즈'가 기사 본문에서 실제로 충족되는지를 판정하라.\n"
            "판정 기준:\n"
            "1) 기사 본문에 클러스터 주제가 직접적으로 해당됨\n"
            "2) 단순 키워드 유사성만으로는 충분치 않음(맥락 기반 판단)\n"
            "3) 단정적 추측/의도 해석 금지, 본문 근거만 반영\n"
            "출력은 JSON 한 줄만: {\\\"match\\\":true|false, \\\"score\\\":0.xx~1.xx, \\\"evidence\\\":\\\"짧은 근거\\\"}\\n\\n"
            f"[cluster]\n{cluster_name}\n\n"
            f"[seed_terms]\n{term_line}\n\n"
            f"[제목]\n{self._clean(title)}\n\n"
            f"[요약]\n{self._clean(summary)}\n\n"
            f"[본문]\n{self._clean(body)[:9000]}"
        )

        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 220,
                },
                timeout=35,
            )
            if r.status_code != 200:
                return False, "judge_api_fail_skip", 0.0

            raw = (r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            m_json = re.search(r"\{[\s\S]*\}", raw)
            if not m_json:
                return False, "judge_parse_fail", 0.0

            data = None
            try:
                data = json.loads(m_json.group(0))
            except Exception:
                return False, "judge_parse_fail", 0.0

            raw_score = str(data.get("score", "")).replace(",", ".")
            try:
                score = float(raw_score)
            except Exception:
                score = 1.0 if str(data.get("match", "false")).lower() in {"true", "1", "yes"} else 0.0

            # if 0~5 scale, normalize to 0~1.
            score = max(0.0, min(1.0, score / 5.0 if score > 1.5 else score))
            matched = bool(data.get("match", False)) if isinstance(data.get("match", False), bool) else str(data.get("match", "")).lower() in {"true", "1", "yes"}
            reason = str(data.get("evidence", "")).strip()
            return matched, reason, score
        except Exception:
            return False, "judge_exception_skip", 0.0

    def _llm_practical_implication(self, title: str, body: str, summary: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return ""
        prompt = (
            "아래 기사 본문/요약을 바탕으로 실무 시사점을 한국어로 작성하라.\n"
            "규칙:\n"
            "1) 정확히 2문장\n"
            "2) 조언/권고/지시 문장 금지(예: ~해야 한다, 필요하다, 권장한다, 점검해야)\n"
            "3) 기사에서 드러나는 핵심 변화와 맥락만 서술\n"
            "4) 상투적 패턴 문구 반복 금지\n"
            "5) 메타 표현 금지(기사에는/본문에서는/실무에서는 등)\n"
            "출력은 시사점 본문만.\n\n"
            f"[제목]\n{title}\n\n[요약]\n{summary}\n\n[본문]\n{body[:8000]}"
        )
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 300,
                },
                timeout=30,
            )
            if r.status_code != 200:
                return ""
            txt = (r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            return self._clean(txt)
        except Exception:
            return ""

    def _llm_generate_title_from_body(self, url_title: str, body: str) -> str:
        base_title = self._clean(url_title)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return base_title

        prompt = (
            "너는 한국어 뉴스레터 에디터다. 기사 본문과 URL 원제목을 바탕으로 한국어 기사 제목을 1줄로 작성하라.\n"
            "규칙:\n"
            "1) 본문 근거 기반\n"
            "2) 과장/추측 금지\n"
            "3) 주체/행동/결과가 드러나게\n"
            "4) 출력은 제목만\n\n"
            f"[URL 원제목]\n{base_title}\n\n"
            f"[기사 본문]\n{body[:5000]}"
        )
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 120,
                },
                timeout=20,
            )
            if r.status_code != 200:
                return base_title
            txt = (r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            return self._clean(txt) or base_title
        except Exception:
            return base_title

    @staticmethod
    def _clean(text: str) -> str:
        t = unescape(text or "")
        t = re.sub(r"\s+", " ", t).strip()
        return t

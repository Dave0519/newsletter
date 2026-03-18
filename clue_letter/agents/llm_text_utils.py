from __future__ import annotations

import json
import os
from typing import List

import requests


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "") or ""


def _base_payload(prompt: str, max_tokens: int = 500, temp: float = 0.2) -> dict:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
    }
    if model.lower().startswith("gpt-5"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    return payload


def _llm(prompt: str, max_tokens: int = 600, temp: float = 0.2) -> str:
    key = _api_key()
    if not key:
        return ""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(OPENAI_API_URL, headers=headers, json=_base_payload(prompt, max_tokens=max_tokens, temp=temp), timeout=60)
        if r.status_code != 200:
            return ""
        data = r.json()
        return ((data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip())
    except Exception:
        return ""


def translate_ko(text: str) -> str:
    if not text:
        return ""
    prompt = (
        "너는 번역 편집자다. 주어진 텍스트를 자연스러운 한국어로 번역해줘.\n"
        "직역이 아닌, 한국어 문장으로 원문의 의미를 훼손하지 않게 정리.\n"
        "기술명, 기업명, 통화 단위, 숫자는 최대한 보존해줘.\n\n"
        f"[원문]\n{text}"
    )
    out = _llm(prompt, max_tokens=1200, temp=0.15)
    return out.strip() if out else text


def summarize_ko(body: str, title: str = "", sentence_count: int = 4) -> str:
    if not body:
        return ""
    count = max(1, min(sentence_count, 6))
    # 요약은 제목이 아니라 원문 텍스트만으로 생성한다.
    prompt = (
        '너는 뉴스 분석 에디터다. 아래 원문에서 외부 추측 없이 핵심만 추출해 한국어로 4~5줄 요약해줘.\n'
        '원문에 없는 내용은 절대 넣지 말고, 숫자/사실/행위 중심으로 가독성 있게 정리해.\n'
        '문장 당 1~2문장, 불필요한 수식어 없이 핵심만.\n\n'
        f"원문:\n{body}"
    )
    out = _llm(prompt, max_tokens=900, temp=0.2)
    if out:
        lines = [x.strip() for x in out.splitlines() if x.strip()]
        return "\n".join(lines[:count])
    # fallback: heuristic 1st lines
    cleaned = " ".join((body or "").split())
    return "\n".join([s.strip() + ("." if not s.strip().endswith(".") else "") for s in cleaned.split(".")[:count] if s.strip()])

def practical_ko(title: str, summary: str, max_sentences: int = 5) -> str:
    if not title and not summary:
        return ""
    prompt = (
        "실무 시사점 작성 원칙에 따라 작성해줘.\n"
        "1) 기사의 주요 사실(출시, 수익, 감원, 계약, 정책 등)과 직접 연결된 실무적 영향으로만 구성\n"
        "2) 기업/팀/개인이 즉시 적용 가능한 조치, 리스크, 전략적 대응 방향을 포함\n"
        "3) 행동 지향 문장을 허용\n"
        "4) 기사 본문/요약 근거 외의 일반 상식·추정·추측은 배제\n"
        "5) 단문 위주 3~5문장 이내\n\n"
        "[요약본]\n"
        f"{summary}\n\n"
        "작성 지침:\n"
        "- 모호한 표현(아마, 추정, 것으로 보인다) 최소화\n"
        "- 조치/모니터링 포인트가 있으면 동사형으로 제시\n"
        "- 항목 번호(1), 2), 첫째, 둘째, • 등 번호/리스트 형식 사용 금지\n"
    )
    out = _llm(prompt, max_tokens=500, temp=0.2)
    if not out:
        return (summary or "")[:max_sentences * 80].strip()

    chunks = [x.strip() for x in out.replace("\n\n", "\n").split("\n") if x.strip()]
    cleaned = []
    import re
    for c in chunks:
        c2 = re.sub(r"^\\s*(?:[0-9]+[\\)\.]|[가-힣]+\\)|첫째|둘째|셋째|넷째|다섯째|여섯째)[\\s:\,-]*", "", c).strip()
        c2 = re.sub(r"\\s+", " ", c2)
        if c2:
            cleaned.append(c2)
    return "\n".join(cleaned[: min(max_sentences, 5)])


def rewrite_title(raw_title: str, body: str, max_chars: int = 70) -> str:
    """원문 제목을 그대로 쓰지 않고 70자 이내 브리핑형 제목으로 다시 작성."""
    if not raw_title:
        return ""

    prompt = (
        '너는 뉴스 헤드라인 편집자다. 아래 기사 내용을 바탕으로, 출처/매체명 없이 공백과 기호 정리를 해서 '
        '"행동/사실" 중심의 짧은 제목 하나만 한국어로 작성해줘.\n'
        '원문 제목은 그대로 반복하지 말고, 핵심 결론을 압축해 다시 쓰기.\n\n'
        f"[원문 제목]\n{raw_title}\n\n"
        f"[본문 일부]\n{body[:1200]}"
    )
    out = _llm(prompt, max_tokens=80, temp=0.2)
    if not out:
        return raw_title[:max_chars]

    s = (out or "").strip().replace("\n", " ")
    # 한두 줄 정도 길이로 정리
    while "  " in s:
        s = s.replace("  ", " ")
    return s[:max_chars]


def extract_hashtags(texts: List[str], top_n: int = 6) -> str:
    if not texts:
        return ""
    from collections import Counter
    # very simple keyword extraction: high-frequency noun-like tokens
    tokens: list[str] = []
    for t in texts:
        for w in (t.replace("/", " ").replace("(", " ").replace(")", " ").split()):
            w = w.strip(" .,;:!?#\"')(").replace("\n", "")
            if len(w) >= 2:
                tokens.append(w)
    c = Counter(tokens)
    items = [k for k, _ in c.most_common(top_n)]
    return " ".join([f"#{x}" for x in items])

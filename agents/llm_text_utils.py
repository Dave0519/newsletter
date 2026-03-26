from __future__ import annotations

import json
import logging
import os
import re
from typing import List

import requests

logger = logging.getLogger(__name__)


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
        logger.warning("llm request skipped: OPENAI_API_KEY missing")
        return ""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = _base_payload(prompt, max_tokens=max_tokens, temp=temp)
    try:
        r = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            snippet = (r.text or "").strip()[:500]
            logger.warning(
                "llm request failed: status=%s snippet=%s",
                r.status_code,
                snippet,
            )
            return ""
        data = r.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        if not content:
            logger.warning("llm request returned empty content: model=%s", payload.get("model"))
        return content
    except requests.RequestException as e:
        logger.warning("llm request failed: network error: %s", e)
        return ""
    except ValueError as e:
        # json decoding 실패
        logger.warning("llm request failed: json decode error: %s", e)
        return ""
    except Exception as e:
        logger.exception("llm request failed: unexpected exception")
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


def summarize_ko(body: str, title: str = "", sentence_count: int = 6) -> str:
    if not body:
        return ""

    def _to_one_paragraph_six_sentences(text: str, count: int = 6) -> str:
        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return ""
        parts = [p.strip() for p in re.split(r"(?<=[\.\!\?。！？])\s+", raw) if p.strip()]
        out: list[str] = []
        for p in parts:
            if not re.search(r"[\.\!\?。！？]$", p):
                p += "."
            out.append(p)
            if len(out) >= count:
                break
        return " ".join(out[:count]).strip()

    prompt = (
        "당신은 뉴스 원문을 정밀하게 압축 요약하는 편집자입니다.\n\n"
        "아래에 제공되는 기사 원문만 읽고, 반드시 한국어로 6문장 요약을 작성하세요.\n"
        "원문이 영어(또는 다국어)여도 출력은 반드시 한국어로만 작성하세요.\n\n"
        "[핵심 원칙]\n"
        "- 반드시 기사 원문에 실제로 있는 내용만 사용하세요.\n"
        "- 외부 지식, 일반 상식, 배경 설명, 추측, 해석, 전망, 평가를 절대 추가하지 마세요.\n"
        "- 원문에 없는 인물, 기관, 원인, 의도, 영향, 맥락을 만들어 넣지 마세요.\n"
        "- 요약은 핵심 정보만 남기고 노이즈 없이 작성하세요.\n"
        "- 숫자, 사실, 행위, 조치, 발표, 변화 내용처럼 검증 가능한 정보 중심으로 정리하세요.\n"
        "- 문장은 짧고 명확하게 쓰되, 리스트처럼 나열하지 말고 자연스러운 기사체 문단으로 작성하세요.\n"
        "- 각 문장은 너무 짧게 끊지 말고, 핵심 사실을 충분히 담아 약간 더 길고 정보 밀도 있게 작성하세요.\n"
        "- 최종 출력은 뉴스레터 본문에서 양쪽정렬로 읽힌다는 점을 고려해 문장 길이와 호흡을 균형 있게 맞추세요.\n"
        "- 기사 문장을 길게 복사하지 말고 의미만 압축해서 재작성하세요.\n"
        "- 회사명, 제품명, 서비스명, 인명 등 고유명사의 영문 표기가 원문에 있으면 가능한 한 원문 표기를 유지하세요(예: SK hynix, NVIDIA DLSS 5, Arm, Meta).\n\n"
        "[반드시 제외할 요소]\n"
        "- 기사 제작일, 입력일, 수집일, 등록일, 작성 시각 등 메타성 날짜 정보\n"
        "- 기자명 / 출처명 / 매체명 / 통신사 표기\n"
        "- 저작권 문구\n"
        "- 기사 메타정보\n"
        "- 광고성 문구\n"
        "- 기사 본문과 무관한 배경 잡설\n"
        "- 원문 밖 해석, 추론, 전망, 평가\n"
        "- 제목 반복\n"
        "- 해시태그, 이모지, 불릿 기호\n\n"
        "[날짜 처리 규칙]\n"
        "- 기사 본문 안에 핵심 사실로 포함된 날짜는 유지할 수 있습니다.\n"
        "- 단, 기사의 메타정보에 해당하는 날짜(예: 작성일, 송고일, 수정일, 입력일)는 절대 포함하지 마세요.\n\n"
        "[출력 형식]\n"
        "- 반드시 6문장만 출력하세요.\n"
        "- 줄바꿈 없이 하나의 문단으로 출력하세요.\n"
        "- 숫자 목록, 불릿, 기호 없이 자연스러운 서술형 문장으로 작성하세요.\n"
        "- 각 문장은 독립적으로 읽혀야 하며, 전체적으로는 하나의 짧은 기사 요약처럼 자연스럽게 이어져야 합니다.\n"
        "- 한 문장에 하나의 핵심 사실만 담으세요.\n"
        "- 중요도 순으로 배열하세요.\n"
        "- 불확실하거나 근거가 약한 내용은 제외하세요.\n"
        "- 원문에 수치가 있으면 가능한 한 유지하세요.\n"
        "- 원문에 서로 다른 행위 주체가 나오면 헷갈리지 않게 분리해서 쓰세요.\n\n"
        "[품질 기준]\n"
        "- 가장 중요한 사실이 먼저 와야 합니다.\n"
        "- 요약문만 읽어도 기사 핵심이 파악되어야 합니다.\n"
        "- 군더더기 없이 정보 밀도가 높아야 합니다.\n"
        "- 문장들이 메모처럼 끊기지 않고 자연스러운 기사체 흐름을 가져야 합니다.\n"
        "- 원문에 없는 내용이 단 하나라도 들어가면 안 됩니다.\n\n"
        "출력 전 스스로 점검하세요.\n"
        "- 원문에 없는 내용이 포함되었는가?\n"
        "- 메타정보(기자명, 출처, 저작권, 작성일 등)가 들어갔는가?\n"
        "- 출력 언어가 한국어로만 작성되었는가?\n"
        "- 6문장을 정확히 지켰는가?\n"
        "- 줄바꿈 없이 하나의 문단으로 작성되었는가?\n"
        "- 리스트처럼 보이지 않는가?\n"
        "- 핵심 사실보다 잡정보가 많지 않은가?\n"
        "문제가 있으면 수정 후 최종본만 출력하세요.\n\n"
        "이제 아래 기사 원문을 요약하세요.\n\n"
        f"기사 원문 :\n{body}"
    )

    out = _llm(prompt, max_tokens=1200, temp=0.2)
    if out:
        return _to_one_paragraph_six_sentences(out, count=6)

    # fallback: heuristic
    cleaned = re.sub(r"\s+", " ", (body or "").strip())
    return _to_one_paragraph_six_sentences(cleaned, count=6)

def summarize_core_ko(body: str, title: str = "", sentence_count: int = 2) -> str:
    if not body:
        return ""

    cnt = max(1, min(sentence_count, 2))
    prompt = (
        "당신은 뉴스 원문을 정밀하게 압축 요약하는 편집자입니다.\n\n"
        "아래 기사 원문만 바탕으로, core_description용 요약을 작성하세요.\n"
        "원문이 영어(또는 다국어)여도 출력은 반드시 한국어로만 작성하세요.\n\n"
        "[핵심 원칙]\n"
        "- 반드시 기사 원문에 실제로 있는 내용만 사용하세요.\n"
        "- 외부 지식, 해석, 추측, 평가, 전망을 추가하지 마세요.\n"
        "- 숫자/사실/행위 중심으로 검증 가능한 정보만 정리하세요.\n"
        "- 메타정보(송고/입력/기자명/출처/저작권/광고)는 제외하세요.\n"
        "- 영어 고유명사(회사명/제품명/서비스명/인명)는 원문 표기를 유지하세요.\n"
        "- 뉴스레터 양쪽정렬 본문에 맞게 문장 길이와 호흡을 균형 있게 작성하세요.\n\n"
        "[출력 형식]\n"
        "- 한국어로만 작성하세요.\n"
        "- 정확히 2문장으로 작성하세요.\n"
        "- 줄바꿈을 넣어 2줄로 출력하세요(문장당 한 줄).\n"
        "- 각 문장은 독립적으로 읽히되, 함께 읽으면 하나의 짧은 핵심 요약이 되게 하세요.\n"
        "- 너무 짧은 메모형 문장 대신 핵심 사실이 충분히 담긴 문장으로 작성하세요.\n"
        "- 리스트 번호/불릿/기호를 붙이지 마세요.\n\n"
        "이제 아래 기사 원문을 요약하세요.\n\n"
        f"기사 원문 :\n{body}"
    )

    out = _llm(prompt, max_tokens=500, temp=0.2)
    if out:
        raw = re.sub(r"\r", "\n", out)
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        if len(lines) >= cnt:
            return "\n".join(lines[:cnt])
        # 줄바꿈이 없으면 문장 분할
        parts = [p.strip() for p in re.split(r"(?<=[\.\!\?。！？])\s+", re.sub(r"\s+", " ", raw)) if p.strip()]
        fixed = []
        for p in parts:
            if not re.search(r"[\.\!\?。！？]$", p):
                p += "."
            fixed.append(p)
            if len(fixed) >= cnt:
                break
        if fixed:
            return "\n".join(fixed[:cnt])

    # fallback
    cleaned = re.sub(r"\s+", " ", (body or "").strip())
    parts = [p.strip() for p in re.split(r"(?<=[\.\!\?。！？])\s+", cleaned) if p.strip()]
    fixed = []
    for p in parts:
        if not re.search(r"[\.\!\?。！？]$", p):
            p += "."
        fixed.append(p)
        if len(fixed) >= cnt:
            break
    return "\n".join(fixed[:cnt])


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
    # 번호/리스트 접두어 제거(패턴 컴파일 오류 방지용 예외 처리 포함).
    bullet_prefix_re = r"^\s*(?:[0-9]+[\)\.]|[가-힣]+[\)]|첫째|둘째|셋째|넷째|다섯째|여섯째)[\s:,\-]*"
    for c in chunks:
        try:
            c2 = re.sub(bullet_prefix_re, "", c).strip()
            c2 = re.sub(r"\s+", " ", c2)
        except Exception:
            c2 = c.strip()
        if c2:
            cleaned.append(c2)
    return "\n".join(cleaned[: min(max_sentences, 5)])



def _has_korean(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7a3]", text or ""))


def _strip_source_noise_from_title(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # remove trailing/leading metadata wrappers like " | Reuters" , " - CNN" , " · Yonhap"
    for sep in [" | ", " - ", " – ", " — ", " : ", " · "]:
        if sep in t:
            left, right = t.rsplit(sep, 1)
            if right and 2 <= len(right) <= 45:
                t = left.strip()
                break

    # remove common metadata suffix patterns
    t = re.sub(r"\s*\|\s*[^|]{1,60}$", "", t)
    t = re.sub(r"\s*-\s*[^-]{1,60}$", "", t)
    t = re.sub(r"\s*·\s*[^·]{1,60}$", "", t)
    t = re.sub(r"\s*\[[^\]]{1,60}\]$", "", t)

    # remove news site/domain fragments
    site_fragments = [
        "Reuters", "AP", "Bloomberg", "Nikkei", "CNBC", "BBC", "KBS", "MBC", "YTN", "SBS",
        "연합뉴스", "조선일보", "중앙일보", "매일경제", "한국경제", "헤럴드", "동아일보", "뉴시스", "연합", "파이낸셜뉴스",
        "the guardian", "the verge", "techcrunch", "ft.com", "reuters.com", "bloomberg.com", "wsj.com", "cnn.com", "yahoo.com", "google", "google news",
    ]
    for s in site_fragments:
        if t.endswith(s):
            t = t[: -len(s)].strip().rstrip("-|:·|·")

    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"^[-\s·:]+|[-\s·:]+$", "", t).strip()
    return t


def rewrite_title(raw_title: str, body: str, max_chars: int = 70) -> str:
    """원문 제목을 그대로 쓰지 않고 70자 이내 브리핑형 제목으로 다시 작성."""
    if not raw_title:
        return ""

    prompt = (
        '너는 뉴스 헤드라인 편집자다. 아래 기사 내용을 바탕으로 출처/매체명/메타데이터를 제거해서 '
        '"행동/사실" 중심의 짧은 제목 하나만 작성해.\n\n'
        '필수 조건:\n'
        '- 제목 전체는 한국어로 작성할 것.\n'
        '- 다만 기업명, 제품명, 프로젝트명, 표준명칭, 고유명사, 약어, 영문 브랜드명/서비스명/모델명은 영문 원문 표기를 유지해도 됨.\n'
        '- 원문 제목에 붙은 메타데이터를 반드시 제거할 것: 기자명, 출처명, 매체명, 뉴스사이트명, 파이프(|), 대시(-), 닷(·), 콜론(:)으로 이어진 접미/접두부.\n'
        '- 특히 "제목 | 출처", "제목 - 뉴스사", "제목 · 매체" 형태는 원천적으로 제거.\n'
        '- 원문 제목을 단순 복붙하지 말고 핵심 결론이 드러나는 문장형 헤드라인으로 재작성.\n'
        '- 한국어 구문이 아니면 번역/의역해 한국어 제목으로 한문장으로 작성.\n'
        '- 70자 이내로 간결하게.\n'
        '- 예시는 금지: 네이버뉴스, 기사요약, 오늘의... 등 메타성 텍스트\n\n'
        f"[원문 제목]\n{raw_title}\n\n"
        f"[본문 일부]\n{body[:1200]}"
    )
    out = _llm(prompt, max_tokens=140, temp=0.15)
    if not out:
        # LLM 실패 시에도 최소한 제목을 한글로 정리해 출력
        fallback = translate_ko(f"{raw_title}\n{body[:400]}")
        s = _strip_source_noise_from_title(fallback or raw_title)
        return s[:max_chars]

    s = (out or "").strip().replace("\n", " ")
    s = _strip_source_noise_from_title(s)

    # 한두 줄 정도 길이로 정리
    while "  " in s:
        s = s.replace("  ", " ")

    # 모델 응답이 한국어가 아닐 경우 보수적으로 번역 보정
    if not _has_korean(s):
        s = translate_ko(s)
        s = _strip_source_noise_from_title(s)

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

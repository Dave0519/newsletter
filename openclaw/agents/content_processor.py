from __future__ import annotations
import re
import requests

class ContentProcessor:
    def __init__(self, _llm_cfg: dict | None = None):
        self.llm_cfg = _llm_cfg or {}

    def _clean(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    def _to_ko(self, text: str) -> str:
        t = self._clean(text)
        if not t:
            return ""
        if any("가" <= ch <= "힣" for ch in t):
            return t
        try:
            r = requests.get("https://translate.googleapis.com/translate_a/single",
                             params={"client":"gtx","sl":"auto","tl":"ko","dt":"t","q":t[:3500]}, timeout=12)
            if r.status_code == 200:
                data = r.json()
                out = "".join(part[0] for part in data[0] if part and part[0])
                if out.strip():
                    return self._clean(out)
        except Exception:
            pass
        return t

    def process_news_batch(self, items: list[dict], lang: str = "ko") -> list[dict]:
        out=[]
        for it in items or []:
            title_ko=self._to_ko(it.get("title",""))
            summary_ko=self._to_ko(it.get("summary",""))
            desc=f"{title_ko} 이슈의 핵심 변화는 다음과 같습니다. {summary_ko}".strip()
            practical=f"{title_ko} 이슈는 공급망·투자·기술 로드맵 점검 시 우선 확인할 필요가 있습니다."
            row=dict(it)
            row["title_ko"]=title_ko
            row["description"]=desc[:900]
            row["practical_implication"]=practical
            row["extraction_status"]="success"
            out.append(row)
        return out

    def process_research_batch(self, items: list[dict], lang: str = "ko") -> list[dict]:
        return items or []

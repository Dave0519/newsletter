from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import List, Sequence

from .models import UserProfile


ALIASES_BY_NEED = {
    "ai 인프라": [
        "AI 인프라",
        "AI infrastructure",
        "AI infra",
        "cloud infrastructure",
        "compute infrastructure",
    ],
    "반도체": [
        "반도체",
        "semiconductor",
        "chip",
        "AI chip",
        "HBM",
        "memory",
    ],
    "데이터센터": [
        "데이터센터",
        "데이터 센터",
        "data center",
        "data-center",
        "datacenter",
        "AI data center",
        "cloud data center",
        "hyperscale",
        "server farm",
    ],
}


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("  ", " ").split())


def _normalize_compact_kor(s: str) -> str:
    return (s or "").strip().replace(" ", "")


class NeedsAgent:
    """사용자 니즈 관리 전용 에이전트."""

    def __init__(self, user_db_path: Path):
        self.user_db_path = Path(user_db_path)
        self.user_db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.user_db_path.exists():
            self.user_db_path.write_text("[]", encoding="utf-8")

    @staticmethod
    def _canonicalize_need(raw: str) -> str:
        return (raw or "").strip()

    @classmethod
    def _expand_aliases(cls, need: str) -> list[str]:
        key = _norm_text(need).replace(" ", "")
        for raw_key, aliases in ALIASES_BY_NEED.items():
            base = _normalize_compact_kor(raw_key).lower()
            if key == base:
                return aliases[:]
            for a in aliases:
                if key == _normalize_compact_kor(a).lower():
                    return aliases[:]

        # fallback: 원입력 보존 + 표기 보정
        expanded = [need]
        if _normalize_compact_kor(need) == "데이터센터" and "데이터 센터" not in expanded:
            expanded.append("데이터 센터")
        return expanded

    @staticmethod
    def _is_valid_term(term: str) -> bool:
        t = (term or "").strip()
        return bool(t and len(t.replace(" ", "")) >= 2)

    @classmethod
    def _is_subsumed_need(cls, source: str, target: str) -> bool:
        s = _normalize_compact_kor(source).lower()
        t = _normalize_compact_kor(target).lower()
        if not s or not t:
            return False
        return s == t or s in t or t in s

    @staticmethod
    def _dedupe_terms(terms: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for t in terms:
            norm = _norm_text(t)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(t.strip())
        return out

    def build_needs_payload(self, interests: list[str]) -> list[dict]:
        seen: set[str] = set()
        payload: list[dict] = []
        for raw_need in interests:
            need = self._canonicalize_need(raw_need)
            if not self._is_valid_term(need):
                continue
            aliases = self._dedupe_terms([a for a in self._expand_aliases(need) if self._is_valid_term(a)])
            aliases_flat = self._dedupe_terms(aliases + [need])
            need_key = _normalize_compact_kor((aliases[0] if aliases else need)).lower()
            # include/subsumption check (예: "ai 인프라" + "AI 인프라" 계열 충돌 정리)
            merged_target = None
            for existing in payload:
                existing_id = str(existing.get("need_id", "")).lower()
                if existing_id and (self._is_subsumed_need(need_key, existing_id) or self._is_subsumed_need(existing_id, need_key)):
                    merged_target = existing
                    break

            if merged_target is not None:
                merged = self._dedupe_terms(list(merged_target.get("aliases", [])) + aliases_flat)
                if merged:
                    merged_target["aliases"] = merged
                    merged_target["aliases_flat"] = self._dedupe_terms(list(merged_target.get("aliases_flat", [])) + aliases_flat)
                if need and need not in merged_target.get("aliases", []):
                    merged_target["aliases"] = self._dedupe_terms(merged_target["aliases"] + [need])
                    merged_target["aliases_flat"] = self._dedupe_terms(merged_target.get("aliases_flat", []) + [need])
                continue

            seen.add(need_key)
            payload.append({
                "need_id": need_key,
                "need_text": need,
                "aliases": aliases,
                "aliases_flat": aliases_flat,
                "is_active": True,
                "is_valid": True,
            })
        return payload

    def _sync_needs_list(self, profile: dict, interests: list[str]) -> dict:
        interests = [x for x in interests if self._is_valid_term(x)]
        payload = self.build_needs_payload(interests)
        profile["needs_list"] = payload
        profile["interests"] = [x.get("need_text") for x in payload]
        return profile

    def _load(self) -> list[dict]:
        return json.loads(self.user_db_path.read_text(encoding="utf-8"))

    def _save(self, data: list[dict]) -> None:
        self.user_db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _new_code(self) -> str:
        data = self._load()
        used = {u["user_code"] for u in data}
        while True:
            token = secrets.token_urlsafe(6).replace("_", "").replace("-", "")[:8].upper()
            if token not in used:
                return token

    def register_user(
        self,
        name: str,
        interests: Sequence[str],
        email: str = "bonggyu1.choi@sk.com",
        exclusions: Sequence[str] | None = None,
        countries: Sequence[str] | None = None,
        user_code: str | None = None,
    ) -> UserProfile:
        name = name.strip()
        if not name:
            raise ValueError("name is required")
        if not interests:
            raise ValueError("interests is required")

        exclusions = list(exclusions or [])
        countries = list(countries or [])
        data = self._load()

        # if user with same name exists and email equal, update interests
        for u in data:
            if u.get("name") == name and u.get("email") == email:
                merged = list(dict.fromkeys([x.strip() for x in (list(u.get("interests", [])) + list(interests)) if x.strip()]))
                u = self._sync_needs_list(u, merged)
                u["exclusions"] = list(dict.fromkeys([x.strip() for x in list(u.get("exclusions", [])) + list(exclusions) if x.strip()]))
                u["countries"] = list(dict.fromkeys([x.strip().upper() for x in list(u.get("countries", [])) + list(countries) if x.strip()]))
                self._save(data)
                return UserProfile(**u)

        uid = (user_code or self._new_code()).upper()
        profile_dict = {
            "name": name,
            "email": email,
            "user_code": uid,
            "interests": [x.strip() for x in interests if x.strip()],
            "exclusions": [x.strip() for x in exclusions if x.strip()],
            "countries": [x.strip().upper() for x in countries if x.strip()],
        }
        profile = self._sync_needs_list(profile_dict, profile_dict.get("interests", []))
        profile = UserProfile(**profile)
        data.append(profile.__dict__)
        self._save(data)
        return profile

    def list_users(self) -> list[UserProfile]:
        return [UserProfile(**u) for u in self._load()]

    def set_user_interests(self, user_code: str, interests: Sequence[str]) -> UserProfile:
        users = self._load()
        for u in users:
            if u["user_code"] == user_code:
                u = self._sync_needs_list(u, list(interests))
                self._save(users)
                return UserProfile(**u)
        raise ValueError(f"user not found: {user_code}")

    def set_user_status(self, user_code: str, active: bool) -> None:
        users = self._load()
        for u in users:
            if u["user_code"] == user_code:
                u["is_active"] = bool(active)
                break
        self._save(users)

    def get_user(self, user_code: str) -> UserProfile:
        for u in self._load():
            if u["user_code"] == user_code:
                return UserProfile(**u)
        raise ValueError(f"user not found: {user_code}")

    def ensure_queries_by_interests(self, interests: List[str], templates: list[str] | None = None) -> List[str]:
        out: list[str] = []
        base_templates = templates or ["{}"]
        for it in interests:
            s = str(it).strip()
            if not s:
                continue
            for t in base_templates:
                out.append(t.format(s))

        # dedupe
        uniq = []
        seen = set()
        for q in out:
            q2 = q.strip()
            if not q2 or q2 in seen:
                continue
            seen.add(q2)
            uniq.append(q2)
        return uniq

    def build_need_queries(self, needs_payload: list[dict], templates: list[str] | None = None) -> list[tuple[str, str]]:
        tpl = templates or ["{}"]; out: list[tuple[str, str]] = []
        for item in needs_payload:
            if not isinstance(item, dict):
                continue
            need_id = str(item.get("need_id", "")).strip()
            need_text = str(item.get("need_text", "")).strip()
            for alias in item.get("aliases", []):
                alias = str(alias).strip()
                if not alias:
                    continue
                for q in self.ensure_queries_by_interests([alias], templates=tpl):
                    out.append((need_id, q))
        deduped: list[tuple[str, str]] = []
        seen = set()
        for need_id, q in out:
            if q in seen:
                continue
            seen.add(q)
            deduped.append((need_id, q))
        return deduped

    def build_need_list_from_user(self, user: UserProfile) -> list[dict]:
        if isinstance(user.needs_list, list) and user.needs_list:
            return user.needs_list
        payload = self.build_needs_payload(user.interests)
        return payload

    def build_user_need_aliases_flat(self, user: UserProfile) -> list[str]:
        alias_set: list[str] = []
        seen = set()
        payload = self.build_need_list_from_user(user)
        for item in payload:
            for alias in item.get("aliases", []):
                a = str(alias).strip()
                if not a or a.lower() in seen:
                    continue
                seen.add(a.lower())
                alias_set.append(a)
        return alias_set

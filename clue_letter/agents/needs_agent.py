from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import List, Sequence

from .models import UserProfile


class NeedsAgent:
    """사용자 니즈 관리 전용 에이전트."""

    def __init__(self, user_db_path: Path):
        self.user_db_path = Path(user_db_path)
        self.user_db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.user_db_path.exists():
            self.user_db_path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        return json.loads(self.user_db_path.read_text(encoding="utf-8"))

    def _save(self, data: list[dict]) -> None:
        self.user_db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _new_code(self) -> str:
        data = self._load()
        used = {u["user_code"] for u in data}
        while True:
            # URL-safe, uppercase-friendly pseudo-random 8 chars
            token = secrets.token_urlsafe(6).replace("_", "").replace("-", "")[:8]
            token = token.upper()
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
                u["interests"] = merged
                u["exclusions"] = list(dict.fromkeys([x.strip() for x in list(u.get("exclusions", [])) + list(exclusions) if x.strip()]))
                u["countries"] = list(dict.fromkeys([x.strip() for x in list(u.get("countries", [])) + list(countries) if x.strip()]))
                self._save(data)
                return UserProfile(**u)

        uid = (user_code or self._new_code()).upper()
        profile = UserProfile(
            name=name,
            email=email,
            user_code=uid,
            interests=[x.strip() for x in interests if x.strip()],
            exclusions=[x.strip() for x in exclusions if x.strip()],
            countries=[x.strip().upper() for x in countries if x.strip()],
        )
        data.append(profile.__dict__)
        self._save(data)
        return profile

    def list_users(self) -> list[UserProfile]:
        return [UserProfile(**u) for u in self._load()]

    def set_user_interests(self, user_code: str, interests: Sequence[str]) -> UserProfile:
        users = self._load()
        for u in users:
            if u["user_code"] == user_code:
                u["interests"] = [x.strip() for x in interests if x.strip()]
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

    def ensure_queries_by_interests(self, interests: List[str]) -> List[str]:
        # 니즈별로 여러 검색 패턴 생성
        out: list[str] = []
        base_templates = [
            "{}",
            "{} news",
            "{} AI",
            "{} 업데이트",
            "{} 현황",
        ]
        for it in interests:
            s = str(it).strip()
            if not s:
                continue
            for t in base_templates:
                out.append(t.format(s))
        # 중복 제거 유지순서
        uniq = []
        seen = set()
        for q in out:
            q2 = q.strip()
            if not q2 or q2 in seen:
                continue
            seen.add(q2)
            uniq.append(q2)
        return uniq

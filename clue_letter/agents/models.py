from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class UserNeed:
    user_code: str
    category: str
    topic: str
    weight: float = 1.0
    queries: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    name: str
    email: str
    user_code: str
    interests: List[str]
    exclusions: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CollectedArticle:
    title: str
    url: str
    country: str
    summary: str
    body: str
    source: str = "browser"
    need_category: Optional[str] = None
    query: Optional[str] = None
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    relevance_note: str = ""
    relevance_score: float = 0.0


@dataclass
class NewsletterEntry:
    title: str
    summary: str
    url: str
    country: str
    practical_implication: str
    need_category: Optional[str] = None
    topic: Optional[str] = None

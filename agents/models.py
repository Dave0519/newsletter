from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class UserNeed:
    need_id: str
    need_text: str
    aliases: List[str]
    aliases_flat: List[str] = field(default_factory=list)
    category: str = "general"
    weight: float = 1.0
    is_active: bool = True
    queries: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    name: str
    email: str
    user_code: str
    interests: List[str]
    needs_list: List[dict] = field(default_factory=list)
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
    source_type: str = "direct"
    need_category: Optional[str] = None
    matched_need_ids: List[str] = field(default_factory=list)
    matched_needs: List[str] = field(default_factory=list)
    matched_aliases: List[str] = field(default_factory=list)
    query: Optional[str] = None
    published_at: str = ""
    diversity_penalty: float = 0.0
    body_len: int = 0
    extracted_at: str = ""
    extraction_status: str = "pending"
    recency_score: float = 0.0
    need_match_score: float = 0.0
    source_score: float = 1.0
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    relevance_note: str = ""
    relevance_score: float = 0.0
    # observability/compat shim fields
    origin_type: Optional[str] = None
    origin_detail: Optional[str] = None
    issue_angle_id: Optional[str] = None


@dataclass
class NewsletterEntry:
    title: str
    summary: str
    url: str
    country: str
    practical_implication: str
    need_category: Optional[str] = None
    topic: Optional[str] = None

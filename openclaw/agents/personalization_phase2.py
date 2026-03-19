from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


STOP_WORDS = {
    "ai", "the", "and", "for", "with", "from", "that", "this", "news", "update", "latest",
    "2026", "2025", "기업", "기사", "관련", "기술", "산업", "발표", "출시",
}

SKHYNIX_ALIASES = {
    "sk hynix", "sk하이닉스", "hynix", "하이닉스",
}

SKHYNIX_SIGNAL_TERMS = {
    "hbm", "dram", "nand", "memory", "semiconductor", "반도체",
    "ai infrastructure", "ai 인프라", "data center", "데이터센터",
    "gpu", "packaging", "advanced packaging", "supply chain", "메모리",
}


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", (value or "").lower())).strip()


def tokenize_text(value: str) -> list[str]:
    text = _norm_text(value)
    tokens = [tok for tok in text.split() if len(tok) >= 2 and tok not in STOP_WORDS]
    return tokens


def get_issue_angle_id(article: dict[str, Any]) -> str:
    existing = (article.get("issue_angle_id") or article.get("issueAngleId") or "").strip()
    if existing:
        return existing

    text = " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("title_from_url") or ""),
            str(article.get("summary") or ""),
            str(article.get("description") or ""),
        ]
    )
    tokens = tokenize_text(text)
    if not tokens:
        return "issue:unknown"

    aliases = [alias for alias in SKHYNIX_ALIASES if alias in _norm_text(text)]
    anchor = aliases[0].replace(" ", "-") if aliases else tokens[0]
    keywords = []
    for token in tokens:
        if token == anchor or token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= 3:
            break
    if not keywords:
        keywords = tokens[:3]
    return "issue:" + ":".join([anchor] + keywords[:3])


def is_skhynix_relevant(article: dict[str, Any]) -> bool:
    text = _norm_text(
        " ".join(
            [
                str(article.get("title") or ""),
                str(article.get("title_from_url") or ""),
                str(article.get("summary") or ""),
                str(article.get("description") or ""),
                str(article.get("article_body") or "")[:1500],
            ]
        )
    )
    if any(alias in text for alias in SKHYNIX_ALIASES):
        return True
    return sum(1 for term in SKHYNIX_SIGNAL_TERMS if term in text) >= 2


def customer_skhynix_affinity(customer: dict[str, Any]) -> float:
    prefs = customer.get("preferences", {}) if isinstance(customer, dict) else {}
    tokens = []
    for key in ("keywords", "focus_topics", "watch_companies", "search_queries"):
        values = prefs.get(key, []) if isinstance(prefs, dict) else []
        tokens.extend([str(v).lower() for v in values if isinstance(v, str)])
    clusters = prefs.get("needClusters", []) if isinstance(prefs, dict) else []
    for cluster in clusters or []:
        if not isinstance(cluster, dict):
            continue
        tokens.extend([str(v).lower() for v in cluster.get("terms", []) if isinstance(v, str)])

    text = " ".join(tokens)
    score = 0.0
    if any(alias in text for alias in SKHYNIX_ALIASES):
        score += 1.5
    score += 0.5 * sum(1 for term in SKHYNIX_SIGNAL_TERMS if term in text)
    return score


def build_customer_query_plans(customer: dict[str, Any], built_queries: list[str]) -> list[dict[str, Any]]:
    prefs = customer.get("preferences", {}) if isinstance(customer, dict) else {}
    explicit = {q.strip().lower() for q in prefs.get("search_queries", []) if isinstance(q, str) and q.strip()}
    cluster_terms = []
    for cluster in prefs.get("needClusters", []) if isinstance(prefs, dict) else []:
        if not isinstance(cluster, dict):
            continue
        weight = float(cluster.get("weight", 1.0) or 1.0)
        for term in cluster.get("terms", []) or []:
            if isinstance(term, str) and term.strip():
                cluster_terms.append((term.strip().lower(), weight, str(cluster.get("name") or "cluster")))

    affinity = customer_skhynix_affinity(customer)
    plans = []
    seen = set()
    for query in built_queries:
        q = (query or "").strip()
        if not q:
            continue
        q_key = q.lower()
        if q_key in seen:
            continue
        seen.add(q_key)
        weight = 1.0
        source_cluster = ""
        if q_key in explicit:
            weight += 1.5
        for term, cluster_weight, cluster_name in cluster_terms:
            if term and term in q_key:
                weight += min(cluster_weight, 2.0)
                source_cluster = source_cluster or cluster_name
        if any(alias in q_key for alias in SKHYNIX_ALIASES | SKHYNIX_SIGNAL_TERMS):
            weight += min(affinity, 2.0)
        plans.append(
            {
                "query": q,
                "weight": round(weight, 2),
                "customerId": customer.get("customer_id", "default"),
                "originType": "query_aware_source",
                "originDetail": {
                    "query": q,
                    "sourceCluster": source_cluster,
                    "affinityScore": affinity,
                },
            }
        )
    return plans


def _parse_published_at(article: dict[str, Any]) -> datetime:
    raw = str(article.get("published_at") or "").strip()
    if not raw:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _freshness_score(article: dict[str, Any], now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    age_hours = max(0.0, (now - _parse_published_at(article)).total_seconds() / 3600.0)
    return math.exp(-age_hours / 24.0)


def choose_shared_hot_topic(
    candidate_pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    customer: dict[str, Any],
    *,
    hot_topic_cap: int = 1,
    relevance_floor: float = 1.5,
) -> dict[str, Any]:
    affinity = customer_skhynix_affinity(customer)
    if affinity < relevance_floor:
        return {"action": "skip", "reason": "low_affinity", "candidate": None}

    selected_ids = {get_issue_angle_id(item) for item in selected}
    selected_shared = sum(1 for item in selected if item.get("origin_type") == "shared_topic_injected")
    if selected_shared >= hot_topic_cap:
        return {"action": "skip", "reason": "hot_topic_cap_reached", "candidate": None}

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidate_pool:
        if not is_skhynix_relevant(item):
            continue
        issue_id = get_issue_angle_id(item)
        clusters[issue_id].append(item)

    best_issue = None
    best_score = -1.0
    best_candidate = None
    for issue_id, items in clusters.items():
        if issue_id in selected_ids:
            continue
        domains = {str(item.get("source") or item.get("url") or "") for item in items}
        support = len(items)
        score = affinity + support + (len(domains) * 0.5) + max(_freshness_score(item) for item in items)
        candidate = max(items, key=lambda item: (float(item.get("_customer_score", 0)), _freshness_score(item)))
        if score > best_score:
            best_score = score
            best_issue = issue_id
            best_candidate = candidate

    if best_candidate is None:
        return {"action": "skip", "reason": "no_hot_topic_candidate", "candidate": None}

    support_count = len(clusters[best_issue]) if best_issue is not None else 0
    injected = dict(best_candidate)
    injected["issue_angle_id"] = best_issue
    injected["origin_type"] = "shared_topic_injected"
    injected["origin_detail"] = {
        "reason": "dynamic_skhynix_hot_topic",
        "affinityScore": affinity,
        "issueAngleId": best_issue,
        "clusterSupport": support_count,
    }
    return {
        "action": "inject",
        "reason": "eligible_hot_topic",
        "candidate": injected,
        "issueAngleId": best_issue,
        "clusterSupport": support_count,
        "affinityScore": affinity,
    }


def rerank_overlap_aware(
    selected: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    *,
    personalization_share_floor: float = 0.75,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not selected:
        return selected, {"reordered": 0, "removedIssueDuplicates": 0}

    target_n = len(selected)
    combined: list[dict[str, Any]] = []
    seen_urls = set()
    for item in list(selected) + list(candidate_pool):
        url = (item.get("url") or item.get("source_url") or "").strip().lower()
        key = url or get_issue_angle_id(item)
        if key in seen_urls:
            continue
        seen_urls.add(key)
        row = dict(item)
        row["issue_angle_id"] = get_issue_angle_id(row)
        combined.append(row)

    chosen: list[dict[str, Any]] = []
    used_issue_ids = set()
    domain_counts: dict[str, int] = defaultdict(int)
    removed_issue_duplicates = 0
    baseline_personalized = max(1, int(round(target_n * personalization_share_floor)))

    def base_score(item: dict[str, Any]) -> float:
        score = float(item.get("_customer_score", 0))
        score += float(item.get("_precheck_score", 0)) * 0.3
        score += _freshness_score(item) * 0.5
        if item.get("origin_type") == "shared_topic_injected":
            score += 0.8
        return score

    while len(chosen) < target_n and combined:
        best_idx = None
        best_value = None
        for idx, item in enumerate(combined):
            issue_id = item.get("issue_angle_id") or get_issue_angle_id(item)
            domain = re.sub(r"^www\.", "", re.sub(r"^https?://", "", str(item.get("url") or item.get("source") or ""))).split("/")[0]
            penalty = 0.0
            if issue_id in used_issue_ids:
                penalty += 100.0
            penalty += max(0, domain_counts[domain] - 0) * 0.5 if domain else 0.0
            value = base_score(item) - penalty
            if best_value is None or value > best_value:
                best_idx = idx
                best_value = value
        if best_idx is None:
            break
        item = combined.pop(best_idx)
        issue_id = item.get("issue_angle_id")
        if issue_id in used_issue_ids:
            removed_issue_duplicates += 1
            continue
        chosen.append(item)
        used_issue_ids.add(issue_id)
        domain = re.sub(r"^www\.", "", re.sub(r"^https?://", "", str(item.get("url") or item.get("source") or ""))).split("/")[0]
        if domain:
            domain_counts[domain] += 1

    personalized = [item for item in chosen if item.get("origin_type") != "shared_topic_injected"]
    if len(personalized) < baseline_personalized:
        for item in selected:
            if item.get("origin_type") == "shared_topic_injected":
                continue
            url = (item.get("url") or item.get("source_url") or "").strip().lower()
            if any((cand.get("url") or cand.get("source_url") or "").strip().lower() == url for cand in chosen):
                continue
            for idx, chosen_item in enumerate(reversed(chosen)):
                if chosen_item.get("origin_type") == "shared_topic_injected":
                    chosen[len(chosen) - idx - 1] = dict(item)
                    break
            personalized = [row for row in chosen if row.get("origin_type") != "shared_topic_injected"]
            if len(personalized) >= baseline_personalized:
                break

    return chosen[:target_n], {
        "reordered": max(0, len(selected) - sum(1 for idx, item in enumerate(chosen[:target_n]) if idx < len(selected) and item == selected[idx])),
        "removedIssueDuplicates": removed_issue_duplicates,
        "personalizationShare": round(len([item for item in chosen[:target_n] if item.get("origin_type") != "shared_topic_injected"]) / max(1, target_n), 2),
    }

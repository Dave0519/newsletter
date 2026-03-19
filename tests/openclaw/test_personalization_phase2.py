from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "openclaw"))

CLUEOrchestrator = import_module("agents.orchestrator").CLUEOrchestrator
phase2 = import_module("agents.personalization_phase2")
build_customer_query_plans = phase2.build_customer_query_plans
choose_shared_hot_topic = phase2.choose_shared_hot_topic
customer_skhynix_affinity = phase2.customer_skhynix_affinity
rerank_overlap_aware = phase2.rerank_overlap_aware


def _customer(customer_id: str, keywords: list[str], queries: list[str], need_terms: list[str], watch_companies: list[str] | None = None) -> dict:
    return {
        "customer_id": customer_id,
        "preferences": {
            "keywords": keywords,
            "search_queries": queries,
            "watch_companies": watch_companies or [],
            "focus_topics": [],
            "needClusters": [{"name": "cluster", "weight": 1.5, "terms": need_terms}],
        },
    }


class _FakeCollector:
    def __init__(self) -> None:
        self.weighted_queries = []

    def collect_all(self, per_category_limit: int = 30) -> list[dict]:
        return [{"title": "Generic AI launch", "summary": "enterprise AI rollout", "url": "https://static.example.com/a"}]

    def collect_custom_queries(self, queries, category="AI_TECH", limit=40, weighted_queries=None, keywords=None):
        self.weighted_queries = list(weighted_queries or [])
        return [
            {
                "title": "SK hynix HBM demand rises",
                "summary": "HBM and memory demand from AI infrastructure climbs",
                "url": "https://query.example.com/hbm",
                "origin_type": "query_aware_source",
                "origin_detail": {"query": self.weighted_queries[0]["query"] if self.weighted_queries else ""},
                "query_weight": self.weighted_queries[0]["weight"] if self.weighted_queries else 1.0,
            }
        ]


class PersonalizationPhase2Test(unittest.TestCase):
    def test_customer_query_plans_weight_high_affinity_terms(self) -> None:
        customer = _customer(
            "high",
            ["HBM", "AI 인프라", "SK hynix"],
            ["SK hynix HBM demand 2026"],
            ["HBM", "반도체", "AI 인프라"],
            watch_companies=["SK hynix"],
        )
        built_queries = ["SK hynix HBM demand 2026", "global AI infrastructure trend"]
        plans = build_customer_query_plans(customer, built_queries)
        self.assertGreater(plans[0]["weight"], plans[1]["weight"])
        self.assertEqual(plans[0]["originType"], "query_aware_source")

    def test_stage_a_master_pool_includes_query_aware_candidates_before_selection(self) -> None:
        orch = CLUEOrchestrator.__new__(CLUEOrchestrator)
        orch.news_collector = _FakeCollector()
        orch._dedupe_by_title_strict = lambda items: items
        orch._dedupe_semantic = lambda items: items
        orch._build_customer_queries = CLUEOrchestrator._build_customer_queries.__get__(orch, CLUEOrchestrator)
        orch._build_customer_query_plans = CLUEOrchestrator._build_customer_query_plans.__get__(orch, CLUEOrchestrator)
        orch._collect_query_aware_stage_a_pool = CLUEOrchestrator._collect_query_aware_stage_a_pool.__get__(orch, CLUEOrchestrator)
        orch._build_stage_a_master_pool = CLUEOrchestrator._build_stage_a_master_pool.__get__(orch, CLUEOrchestrator)

        policy = {
            "phase2_enabled": True,
            "query_aware_upstream_pool_enabled": True,
            "query_aware_stage_a_meta_limit": 20,
            "per_category_limit": 20,
        }
        customer = _customer(
            "high",
            ["HBM", "AI 인프라", "SK hynix"],
            ["SK hynix HBM demand 2026"],
            ["HBM", "반도체", "AI 인프라"],
            watch_companies=["SK hynix"],
        )
        raw_pool, article_pool, meta = orch._build_stage_a_master_pool(policy, [customer])
        self.assertGreater(meta["queryAwareCount"], 0)
        self.assertEqual(meta["staticCount"], 1)
        self.assertTrue(any(item.get("origin_type") == "query_aware_source" for item in raw_pool))
        self.assertTrue(any(item.get("origin_type") == "query_aware_source" for item in article_pool))

    def test_shared_hot_topic_injects_for_high_affinity_user(self) -> None:
        customer = _customer(
            "high",
            ["HBM", "AI 인프라", "SK hynix"],
            ["SK hynix HBM demand 2026"],
            ["HBM", "반도체", "AI 인프라"],
            watch_companies=["SK hynix"],
        )
        selected = [{"title": "AI chip trend", "summary": "general semiconductor update", "url": "https://a.com/1", "_customer_score": 5}]
        candidate_pool = [
            {"title": "SK hynix HBM win", "summary": "HBM demand surges with AI infrastructure", "url": "https://b.com/1", "source": "rss", "_customer_score": 7},
            {"title": "SK hynix HBM packaging", "summary": "memory packaging and AI infrastructure expansion", "url": "https://c.com/2", "source": "rss", "_customer_score": 6},
        ]
        decision = choose_shared_hot_topic(candidate_pool, selected, customer, hot_topic_cap=1, relevance_floor=1.5)
        self.assertEqual(decision["action"], "inject")
        self.assertEqual(decision["candidate"]["origin_type"], "shared_topic_injected")

    def test_shared_hot_topic_skips_low_affinity_and_saturated_issue(self) -> None:
        low_affinity_customer = _customer("low", ["OpenAI", "Claude"], ["OpenAI model update 2026"], ["OpenAI", "Claude"])
        candidate_pool = [{"title": "SK hynix HBM win", "summary": "HBM demand surges with AI infrastructure", "url": "https://b.com/1", "source": "rss", "_customer_score": 7}]
        decision = choose_shared_hot_topic(candidate_pool, [], low_affinity_customer, hot_topic_cap=1, relevance_floor=1.5)
        self.assertEqual(decision["action"], "skip")
        self.assertEqual(decision["reason"], "low_affinity")

        high_affinity_customer = _customer("high", ["HBM", "SK hynix"], ["SK hynix HBM demand 2026"], ["HBM", "반도체"], watch_companies=["SK hynix"])
        selected = [{"title": "SK hynix HBM win", "summary": "HBM demand surges with AI infrastructure", "url": "https://b.com/1", "issue_angle_id": "issue:sk-hynix:hbm:ai:memory"}]
        candidate_pool[0]["issue_angle_id"] = "issue:sk-hynix:hbm:ai:memory"
        decision = choose_shared_hot_topic(candidate_pool, selected, high_affinity_customer, hot_topic_cap=1, relevance_floor=1.5)
        self.assertEqual(decision["action"], "skip")

    def test_overlap_aware_rerank_reduces_redundancy_and_preserves_personalization_share(self) -> None:
        selected = [
            {"title": "HBM demand rises", "summary": "same event angle one", "url": "https://a.com/1", "issue_angle_id": "issue:1", "_customer_score": 10, "origin_type": "query_aware_source"},
            {"title": "HBM demand rises again", "summary": "same event angle two", "url": "https://a.com/2", "issue_angle_id": "issue:1", "_customer_score": 9, "origin_type": "query_aware_source"},
            {"title": "Different enterprise AI story", "summary": "another issue", "url": "https://b.com/3", "issue_angle_id": "issue:2", "_customer_score": 8, "origin_type": "query_aware_source"},
            {"title": "Shared topic item", "summary": "sk hynix related topic", "url": "https://c.com/4", "issue_angle_id": "issue:3", "_customer_score": 6, "origin_type": "shared_topic_injected"},
        ]
        candidate_pool = selected + [
            {"title": "Fresh unique story", "summary": "new issue angle", "url": "https://d.com/5", "issue_angle_id": "issue:4", "_customer_score": 7, "origin_type": "query_aware_source"}
        ]
        reranked, meta = rerank_overlap_aware(selected, candidate_pool, personalization_share_floor=0.75)
        self.assertEqual(len({item["issue_angle_id"] for item in reranked}), len(reranked))
        self.assertGreaterEqual(meta["personalizationShare"], 0.75)

    def test_customer_skhynix_affinity_distinguishes_high_and_low_affinity(self) -> None:
        high = _customer("high", ["HBM", "SK hynix", "AI 인프라"], ["SK hynix HBM demand 2026"], ["HBM", "반도체"], watch_companies=["SK hynix"])
        low = _customer("low", ["OpenAI", "Claude"], ["OpenAI model update 2026"], ["OpenAI", "Claude"])
        self.assertGreater(customer_skhynix_affinity(high), customer_skhynix_affinity(low))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from dotenv import load_dotenv

from agents.content_processor import ContentProcessor
from agents.country_tagger import CountryTagger
from agents.delivery import DeliveryManager
from agents.news_collector import NewsCollector
from agents.newsletter_builder import NewsletterBuilder
from agents.research_collector import ResearchCollector
from utils.dedup import DedupDB
from utils.logger import setup_logger


class CLUEOrchestrator:
    def __init__(self, base_dir: str):
        load_dotenv(os.path.join(base_dir, ".env"))
        self.log = setup_logger()
        self.base_dir = base_dir
        self.config = self._resolve_env_vars(self._load_yaml("config/config.yaml"))
        self.sources = self._load_yaml("config/sources.yaml")
        self.entities = self._load_yaml("config/country_entities.yaml")
        self.category_rules = self._load_yaml("config/category_rules.yaml")
        self.forbidden_domains = self._load_forbidden_domains()
        self._validate_and_filter_sources()
        self.customers = self._load_customers()

        self.news_collector = NewsCollector(
            self.sources,
            freshness_hours=self.config["news"]["global_scan"]["freshness_hours"],
            forbidden_domains=self.forbidden_domains,
        )
        self.research_collector = ResearchCollector(
            self.sources, freshness_days=self.config["news"]["research_insight"]["freshness_days"]
        )
        self.country_tagger = CountryTagger(self.entities)
        self.processor = ContentProcessor(self.config.get("llm", {}))
        self.builder = NewsletterBuilder(
            template_path=os.path.join(base_dir, "../templates/clue/CLUE_TEMPLATE_OFFICIAL.html"),
            country_order=self.config["news"].get("country_order", ["KR", "US", "CN", "TW", "GLOBAL"]),
        )
        self.delivery = DeliveryManager(self.config["email"], self.config.get("telegram"))
        self.dedup = DedupDB(os.path.join(base_dir, "data/clue.db"))

    def run(self, customer_id: str = "default", email_recipient: Optional[str] = None, dry_run: bool = False):
        """Single-customer pipeline (B~E)."""
        issue_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y.%m.%d")
        customer = self._get_customer(customer_id)

        publish_enabled = self._is_publish_enabled()
        if not publish_enabled:
            result = {
                "status": "blocked",
                "issue_date": issue_date,
                "delivery": {"email": False, "telegram": False},
                "active_countries": [],
                "total_scan": 0,
                "total_research": 0,
                "extraction_stats": {},
                "gate_ok": False,
                "gate_errors": ["publish_switch_disabled"],
                "template_path": str(Path(self.base_dir, "..", "templates", "clue", "CLUE_TEMPLATE_OFFICIAL.html")),
                "template_sha256": "",
                "html": "",
            }
            self._append_run_log(customer_id, result)
            self._append_audit_log(customer_id, result)
            return result

        policy = self._load_collection_policy()

        # Stage A: shared collection (single-customer execution keeps it local)
        article_pool = self.news_collector.collect_all(per_category_limit=policy["per_category_limit"])
        article_pool = self._dedupe_by_title_strict(article_pool)
        article_pool = self._dedupe_semantic(article_pool)

        return self._run_customer_from_pool(
            customer=customer,
            article_pool=article_pool,
            issue_date=issue_date,
            dry_run=dry_run,
            email_recipient=email_recipient,
            policy=policy,
        )

    def run_all_customers(self, dry_run: bool = False, resume: bool = True, time_budget_minutes: int = 110) -> dict:
        """Stage-split batch pipeline with checkpoints and resume support."""
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        issue_date = now.strftime("%Y.%m.%d")
        state = self._load_state() if resume else {}

        if not state or state.get("stage") == "DONE" or state.get("issue_date") != issue_date:
            run_id = f"run-{now.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
            self.log.info(f"RUN_START run_id={run_id} issue_date={issue_date}")
            customer_ids = [c.get("customer_id") for c in self.customers if c.get("enabled", True)]
            state = {
                "run_id": run_id,
                "issue_date": issue_date,
                "started_at": now.isoformat(),
                "stage": "A",
                "customers": customer_ids,
                "completed_customers": [],
                "failed_customers": [],
                "customer_stage": {},
                "paths": {"checkpoints_dir": str(self._checkpoint_dir(run_id))},
                "updated_at": now.isoformat(),
            }
            self._save_state(state)

        if not self._is_publish_enabled():
            state.update({
                "stage": "DONE",
                "updated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "status": "blocked",
                "reason": "publish_switch_disabled",
            })
            self._save_state(state)
            return {"status": "blocked", "reason": "publish_switch_disabled", "run_id": state.get("run_id")}

        policy = self._load_collection_policy()
        start_ts = datetime.now(ZoneInfo("Asia/Seoul")).timestamp()
        deadline = start_ts + max(10, int(time_budget_minutes)) * 60

        # Stage A: shared pool
        if state.get("stage") == "A":
            self.log.info("A_START")
            pool = self.news_collector.collect_all(per_category_limit=policy["per_category_limit"])
            pool = self._dedupe_by_title_strict(pool)
            pool = self._dedupe_semantic(pool)
            master_path = self._checkpoint_dir(state["run_id"]) / "master_candidate_pool.json"
            self._dump_json(master_path, pool)
            state["paths"]["master_candidate_pool"] = str(master_path)
            self.log.info("A_DONE")
            state["stage"] = "B"
            state["updated_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
            self._save_state(state)

        pool_path = state.get("paths", {}).get("master_candidate_pool", "")
        try:
            pool = json.loads(Path(pool_path).read_text(encoding="utf-8")) if pool_path and os.path.exists(pool_path) else []
        except Exception:
            pool = []

        results = []
        for cid in state.get("customers", []):
            if cid in set(state.get("completed_customers", [])):
                continue
            if datetime.now(ZoneInfo("Asia/Seoul")).timestamp() >= deadline:
                break

            customer = self._get_customer(cid)
            self.log.info(f"CUSTOMER_START cid={cid}")
            try:
                result = self._run_customer_from_pool(
                    customer=customer,
                    article_pool=list(pool),
                    issue_date=issue_date,
                    dry_run=dry_run,
                    email_recipient=None,
                    policy=policy,
                    run_id=state.get("run_id"),
                    checkpoint_state=state,
                )
                state["completed_customers"].append(cid)
                state["customer_stage"][cid] = "DONE"
                self.log.info(f"CUSTOMER_DONE cid={cid} status={result.get('status')}")
                results.append({"customer_id": cid, "status": result.get("status"), "gate_ok": result.get("gate_ok")})
            except Exception as e:
                state["failed_customers"].append({"id": cid, "stage": state.get("customer_stage", {}).get(cid, "UNKNOWN"), "reason": str(e)[:400]})
                state["customer_stage"][cid] = "FAILED"
                self.log.warning(f"CUSTOMER_FAIL cid={cid} stage={state.get('customer_stage', {}).get(cid, 'UNKNOWN')} reason={str(e)[:200]}")
            state["updated_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
            self._save_state(state)

        all_done = len(state.get("completed_customers", [])) >= len(state.get("customers", []))
        if all_done:
            state["stage"] = "DONE"
            state["status"] = "partial_failed" if state.get("failed_customers") else "completed"
        else:
            state["stage"] = "B"
            state["status"] = "in_progress"
        state["updated_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        self._save_state(state)

        return {
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "completed": len(state.get("completed_customers", [])),
            "failed": len(state.get("failed_customers", [])),
            "total": len(state.get("customers", [])),
            "results": results,
        }

    def _run_customer_from_pool(
        self,
        customer: dict,
        article_pool: list[dict],
        issue_date: str,
        dry_run: bool,
        email_recipient: Optional[str],
        policy: dict,
        run_id: Optional[str] = None,
        checkpoint_state: Optional[dict] = None,
    ) -> dict:
        customer_id = customer.get("customer_id", "default")

        for a in article_pool:
            a["country"] = self.country_tagger.tag(a.get("title", ""), a.get("summary", ""))
            cat, score = self._infer_category(a)
            a["category"] = cat
            a["_category_score"] = score

        # Stage B: shortlist
        self.log.info(f"B_START cid={customer_id}")
        shortlist_meta = self._select_for_customer(customer, article_pool)
        shortlist_meta = shortlist_meta[: policy["shortlist_meta_cap_per_customer"]]
        if checkpoint_state is not None and run_id:
            checkpoint_state["customer_stage"][customer_id] = "B"
            p = self._checkpoint_dir(run_id) / f"shortlist_{customer_id}.json"
            self._dump_json(p, shortlist_meta)
            checkpoint_state.setdefault("paths", {}).setdefault("shortlists", {})[customer_id] = str(p)
        self.log.info(f"B_DONE cid={customer_id} n={len(shortlist_meta)}")

        # Stage C: precheck (fast filter before expensive LLM steps)
        self.log.info(f"C_START cid={customer_id}")
        prechecked = self._precheck_candidates(customer, shortlist_meta)
        if checkpoint_state is not None and run_id:
            checkpoint_state["customer_stage"][customer_id] = "C"
            p = self._checkpoint_dir(run_id) / f"prechecked_{customer_id}.json"
            self._dump_json(p, prechecked)
            checkpoint_state.setdefault("paths", {}).setdefault("prechecked", {})[customer_id] = str(p)
        self.log.info(f"C_DONE cid={customer_id} n={len(prechecked)}")

        # Stage D: origin read + judge
        self.log.info(f"D_START cid={customer_id}")
        origin_candidates = prechecked[: policy["origin_read_cap_per_customer"]]
        clean_items = self._process_origin_candidates_parallel(
            origin_candidates,
            max_workers=policy.get("d_stage_parallel_workers", 2),
            timeout_sec=policy.get("d_stage_timeout_sec", 1800),
        )
        clean_items = self._dedupe_by_title_strict(clean_items)
        clean_items = self._dedupe_semantic(clean_items)
        if checkpoint_state is not None and run_id:
            checkpoint_state["customer_stage"][customer_id] = "D"
            p = self._checkpoint_dir(run_id) / f"cleaned_{customer_id}.json"
            self._dump_json(p, clean_items)
            checkpoint_state.setdefault("paths", {}).setdefault("cleaned", {})[customer_id] = str(p)
        self.log.info(f"D_DONE cid={customer_id} n={len(clean_items)}")

        # Stage E: shortage-aware refill loop (need/article/country gaps)
        self.log.info(f"E_START cid={customer_id}")
        selected = self._select_for_customer(customer, clean_items)
        candidate_pool = list(clean_items)

        min_articles = int(policy.get("min_articles_hard", policy.get("min_articles_soft", 8)))
        min_countries = int(policy.get("min_country_sections_hard", policy.get("min_country_sections_soft", 2)))
        max_refill_rounds = int(policy.get("refill_max_rounds", 3))

        for rr in range(1, max_refill_rounds + 1):
            coverage = self._evaluate_need_coverage(customer, selected)
            country_count = len(set([it.get("country", "GLOBAL") for it in selected if it.get("country")]))
            article_gap = max(0, min_articles - len(selected))
            country_gap = max(0, min_countries - country_count)
            need_gap = bool(coverage.get("need_gap"))

            self.log.info(
                f"E_GAP cid={customer_id} round={rr} need_gap={need_gap} article_gap={article_gap} country_gap={country_gap}"
            )

            if not need_gap and article_gap == 0 and country_gap == 0:
                break

            cluster_gaps = list((coverage.get("cluster_gap", {}) or {}).items())
            # Process each unmet need cluster separately to avoid diluting one weak cluster with others.
            for c_name, c_data in cluster_gaps:
                gap_queries = self._build_gap_queries_for_cluster(customer, c_name, c_data)
                if not gap_queries:
                    continue

                self.log.info(f"E_CLUSTER_GAP cid={customer_id} round={rr} cluster={c_name} terms={gap_queries}")
                refill_queries = self._build_ko_en_query_pairs(gap_queries)
                if not refill_queries:
                    continue

                refill_meta = self.news_collector.collect_custom_queries(
                    refill_queries,
                    category="CUSTOM_NEEDS",
                    limit=policy["refill_meta_limit"],
                )
                refill_meta = self._dedupe_semantic(refill_meta)
                refill_origin = refill_meta[: policy["refill_origin_cap_per_customer"]]
                refill_clean = self.processor.process_news_batch(refill_origin, lang="ko")
                candidate_pool = self._dedupe_semantic(candidate_pool + refill_clean)
                selected = self._dedupe_semantic(selected + refill_clean)
                selected = self._select_for_customer(customer, selected)

            if country_gap > 0:
                country_queries = self._build_country_gap_queries(customer, selected, country_gap)
                if country_queries:
                    self.log.info(f"E_COUNTRY_GAP cid={customer_id} round={rr} countries={country_queries}")
                    refill_queries = self._build_ko_en_query_pairs(country_queries)
                    if refill_queries:
                        country_meta = self.news_collector.collect_custom_queries(
                            refill_queries,
                            category="CUSTOM_NEEDS",
                            limit=policy["refill_meta_limit"],
                        )
                        country_meta = self._dedupe_semantic(country_meta)
                        country_origin = country_meta[: policy["refill_origin_cap_per_customer"]]
                        country_clean = self.processor.process_news_batch(country_origin, lang="ko")
                        candidate_pool = self._dedupe_semantic(candidate_pool + country_clean)
                        selected = self._dedupe_semantic(selected + country_clean)
                        selected = self._select_for_customer(customer, selected)

        selected = self._rebalance_domain_bias(selected, max_share=policy["domain_max_share"])
        selected = self._apply_country_floor(
            selected=selected,
            fallback_pool=candidate_pool,
            country_floor=policy.get("country_floor", {}),
        )
        selected = self._fill_minimum_articles(selected, candidate_pool, min_articles=min_articles)
        if checkpoint_state is not None and run_id:
            checkpoint_state["customer_stage"][customer_id] = "E"
            p = self._checkpoint_dir(run_id) / f"final_{customer_id}.json"
            self._dump_json(p, selected)
            checkpoint_state.setdefault("paths", {}).setdefault("final_candidates", {})[customer_id] = str(p)
        self.log.info(f"E_DONE cid={customer_id} n={len(selected)}")

        # Stage F: render & deliver
        self.log.info(f"F_START cid={customer_id}")
        research_cfg = self.config["news"].get("research_insight", {})
        research_enabled = bool(research_cfg.get("enabled", True))
        raw_research = self.research_collector.collect(target_count=max(12, research_cfg.get("target_per_newsletter", 2) * 4)) if research_enabled else []
        filtered_research = self._select_research_for_customer(customer, raw_research) if raw_research else []

        scan_by_country = self._group_by_country(selected)
        ok, errors = self.builder.validate(
            scan_by_country,
            filtered_research,
            min_scan=self.config["news"]["global_scan"].get("target_per_newsletter", 10),
            min_research=research_cfg.get("target_per_newsletter", 0),
        )
        if not ok:
            self.log.warning(f"Validation failed: {errors}")

        max_per_country = self.config["news"]["global_scan"].get("max_per_country", 5)
        processed_scan = {c: items[:max_per_country] for c, items in scan_by_country.items()}
        processed_scan = self._enforce_title_summary_consistency(processed_scan)
        # Practical implication generation is deferred to final stage after article selection.
        processed_scan = self.processor.generate_practical_implications(processed_scan)

        max_research = research_cfg.get("max_items", 0)
        processed_research = self.processor.process_research_batch(filtered_research, lang="ko")[:max_research] if (research_enabled and max_research > 0 and filtered_research) else []

        template_path, template_sha256 = self.builder.template_fingerprint()
        hashtags = self._build_needs_hashtags(customer)
        html = self.builder.build(
            processed_scan,
            processed_research,
            issue_date,
            serial_number=customer.get("serial_number", ""),
            needs_hashtags=hashtags,
            brand_mark="SK hynix",
            global_scan_intro="개인 관심사에 맞춘 글로벌 핵심 동향을 엄선해 전해드립니다.",
        )

        gate_ok, hard_errors, soft_warnings = self._validate_pre_send(
            customer_id=customer_id,
            issue_date=issue_date,
            processed_scan=processed_scan,
            html=html,
            template_path=template_path,
            prior_errors=errors,
        )

        delivery_result = {"email": False, "telegram": False}
        if not dry_run and gate_ok:
            recipient = email_recipient or customer.get("email") or self.config["email"]["recipient"]
            delivery_result = self.delivery.deliver(html, issue_date, recipient)
        elif not gate_ok:
            self.log.warning(f"Send blocked by hard gates: {hard_errors}")

        if soft_warnings:
            self.log.warning(f"Soft gate warnings: {soft_warnings}")

        extraction_stats = {}
        for c, items in processed_scan.items():
            ok_cnt = sum(1 for i in items if i.get("extraction_status") == "success")
            extraction_stats[c] = {"success": ok_cnt}

        result = {
            "status": "ok" if gate_ok else "blocked",
            "issue_date": issue_date,
            "delivery": delivery_result,
            "active_countries": [c for c, v in processed_scan.items() if v],
            "total_scan": sum(len(v) for v in processed_scan.values()),
            "total_research": len(processed_research),
            "extraction_stats": extraction_stats,
            "gate_ok": gate_ok,
            "gate_errors": hard_errors,
            "soft_warnings": soft_warnings,
            "template_path": template_path,
            "template_sha256": template_sha256,
            "coverage": self._evaluate_need_coverage(customer, selected),
            "html": html,
        }

        self._append_run_log(customer_id, result)
        self._append_audit_log(customer_id, result)
        self.log.info(f"F_DONE cid={customer_id} status={result.get('status')} gate_ok={result.get('gate_ok')}")
        if checkpoint_state is not None:
            checkpoint_state["customer_stage"][customer_id] = "F"
        return result

    def _state_path(self) -> Path:
        return Path(self.base_dir, "..", "newsletter_agent_state.json")

    def _checkpoint_dir(self, run_id: str) -> Path:
        return Path(self.base_dir, "data", "checkpoints", run_id)

    def _load_state(self) -> dict:
        p = self._state_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: dict):
        p = self._state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")

    def _dump_json(self, path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")

    def _load_collection_policy(self) -> dict:
        cfg_path = os.path.join(self.base_dir, "..", "newsletter_agent.json")
        defaults = {
            "per_category_limit": 150,
            "shortlist_meta_cap_per_customer": 80,
            "origin_read_cap_per_customer": 40,
            "refill_meta_limit": 120,
            "refill_origin_cap_per_customer": 15,
            "domain_max_share": 0.30,
            "d_stage_parallel_workers": 2,
            "d_stage_timeout_sec": 1800,
            "min_articles_soft": 8,
            "min_country_sections_soft": 2,
            "refill_max_rounds": 3,
            "min_articles_hard": 8,
            "min_country_sections_hard": 2,
            "country_floor": {"US": 2, "KR": 1, "CN": 1, "GLOBAL": 1},
        }
        try:
            if not os.path.exists(cfg_path):
                return defaults
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            col = cfg.get("collection", {}) if isinstance(cfg, dict) else {}
            defaults["per_category_limit"] = int(col.get("perCategoryLimit", defaults["per_category_limit"]))
            defaults["shortlist_meta_cap_per_customer"] = int(col.get("shortlistMetaCapPerCustomer", defaults["shortlist_meta_cap_per_customer"]))
            defaults["origin_read_cap_per_customer"] = int(col.get("originReadCapPerCustomer", defaults["origin_read_cap_per_customer"]))
            defaults["refill_meta_limit"] = int(col.get("refillMetaLimit", defaults["refill_meta_limit"]))
            defaults["refill_origin_cap_per_customer"] = int(col.get("refillOriginCapPerCustomer", defaults["refill_origin_cap_per_customer"]))
            defaults["domain_max_share"] = float(col.get("domainMaxShare", defaults["domain_max_share"]))
            defaults["d_stage_parallel_workers"] = int(col.get("dStageParallelWorkers", defaults["d_stage_parallel_workers"]))
            defaults["d_stage_timeout_sec"] = int(col.get("dStageTimeoutSec", defaults["d_stage_timeout_sec"]))
            defaults["min_articles_soft"] = int(col.get("softMinArticlesPerNewsletter", defaults["min_articles_soft"]))
            defaults["min_country_sections_soft"] = int(
                cfg.get("consistencyValidation", {}).get("softMinCountrySections", defaults["min_country_sections_soft"])
            )
            defaults["refill_max_rounds"] = int(col.get("refillMaxRounds", defaults["refill_max_rounds"]))
            defaults["min_articles_hard"] = int(col.get("minArticlesHard", defaults["min_articles_hard"]))
            defaults["min_country_sections_hard"] = int(col.get("minCountrySectionsHard", defaults["min_country_sections_hard"]))
            cf = col.get("countryFloor", defaults["country_floor"])
            if isinstance(cf, dict):
                defaults["country_floor"] = {str(k): int(v) for k, v in cf.items() if str(k) and int(v) >= 0}

        except Exception:
            return defaults
        return defaults

    def _process_origin_candidates_parallel(self, items: list[dict], max_workers: int = 2, timeout_sec: int = 1800) -> list[dict]:
        """Run heavy origin-read + generation in bounded parallel chunks for Stage D."""
        if not items:
            return []

        workers = max(1, int(max_workers or 1))
        chunk_size = 5
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        if workers == 1 or len(chunks) == 1:
            out = []
            for ch in chunks:
                out.extend(self.processor.process_news_batch(ch, lang="ko"))
            return out

        out = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self.processor.process_news_batch, ch, "ko") for ch in chunks]
            try:
                for f in as_completed(futs, timeout=max(30, int(timeout_sec))):
                    try:
                        out.extend(f.result() or [])
                    except Exception as e:
                        self.log.warning(f"D_CHUNK_FAIL: {type(e).__name__}")
            except Exception:
                self.log.warning("D_TIMEOUT: stage-d parallel processing timed out; returning partial results")
        return out

    def _precheck_candidates(self, customer: dict, items: list[dict]) -> list[dict]:
        """Fast precheck before expensive body extraction/LLM generation."""
        prefs = customer.get("preferences", {}) if isinstance(customer, dict) else {}
        clusters = self._extract_need_clusters(customer)
        keywords = []
        for c in clusters:
            keywords.extend(c.get("terms", []) or [])
        if not keywords:
            keywords = [k.strip().lower() for k in (prefs.get("keywords", []) or []) if isinstance(k, str) and k.strip()]

        excludes = [x.strip().lower() for x in (prefs.get("excludes", []) or []) if isinstance(x, str) and x.strip()]

        out = []
        seen_urls = set()
        for it in items or []:
            url = (it.get("url") or "").strip()
            if not url:
                continue
            kurl = url.lower()
            if kurl in seen_urls or self._is_forbidden_url(url):
                continue

            text = f"{it.get('title','')} {it.get('summary','')}".lower()
            if excludes and any(x in text for x in excludes):
                continue

            # keyword-based relevance score (cheap)
            score = sum(text.count(k.lower()) for k in keywords) if keywords else 0
            # keep when matched, or when shortlist is small and category score is non-negative
            if score <= 0 and len(items) > 30 and it.get("_category_score", 0) < 1:
                continue

            row = dict(it)
            row["_precheck_score"] = score
            out.append(row)
            seen_urls.add(kurl)

        out.sort(key=lambda x: (x.get("_precheck_score", 0), x.get("_customer_score", 0), x.get("published_at", "")), reverse=True)
        return out

    def _extract_need_clusters(self, customer: dict) -> list[dict]:
        prefs = customer.get("preferences", {}) if isinstance(customer, dict) else {}
        clusters = prefs.get("needClusters", []) if isinstance(prefs, dict) else []
        if isinstance(clusters, list) and clusters:
            return [c for c in clusters if isinstance(c, dict) and c.get("name")]

        # fallback: focus_topics-driven pseudo clusters
        fallback_terms = [x for x in prefs.get("focus_topics", []) if isinstance(x, str) and x.strip()]
        fallback_keywords = [x for x in prefs.get("keywords", []) if isinstance(x, str) and x.strip()]
        if fallback_terms:
            return [{"name": "focus", "minArticles": 2, "weight": 1.0, "terms": fallback_terms[:8]}]
        return [{"name": "keywords", "minArticles": 1, "weight": 1.0, "terms": fallback_keywords[:10]}]

    @staticmethod
    def _cluster_term_hits(article: dict, term: str) -> int:
        t = (term or "").strip().lower()
        if not t:
            return 0
        if re.search(r"[a-z0-9]", t):
            pattern = rf"\b{re.escape(t)}\b"
        else:
            pattern = re.escape(t)
        text = f"{article.get('title','')} {article.get('summary','')} {article.get('description','')} {article.get('article_body','')[:1500]}".lower()
        return len(re.findall(pattern, text))

    def _evaluate_need_coverage(self, customer: dict, items: list[dict]) -> dict:
        clusters = self._extract_need_clusters(customer)
        cluster_hits = {}
        cluster_gap = {}

        def cluster_score(cluster: dict) -> tuple:
            terms = [x for x in cluster.get("terms", []) if isinstance(x, str) and x.strip()]
            min_required = int(cluster.get("minArticles", 0) or 1)
            weight = float(cluster.get("weight", 1.0) or 1.0)
            hits = {}
            hit_sum = 0
            for t in terms:
                c = self._cluster_term_hits_dict(items, t)
                hits[t] = c
                hit_sum += c
            hit_items = sum(1 for v in hits.values() if v >= 1)
            met = hit_items >= max(1, min(min_required, max(1, len(terms))))
            return cluster.get("name", "cluster"), {
                "minArticles": min_required,
                "weight": weight,
                "terms": terms,
                "term_hits": hits,
                "hit_items": hit_items,
                "hit_total": hit_sum,
                "met": met,
            }

        critical_hits = {}
        for c_name, c_data in map(cluster_score, clusters):
            gap = int(c_data.get("minArticles", 0)) - int(c_data.get("hit_items", 0))
            c_data["gap"] = max(0, gap)
            cluster_hits[c_name] = c_data
            if c_data.get("gap", 0) > 0:
                cluster_gap[c_name] = c_data

        # backward-compatible legacy fields for monitoring compatibility
        flat_critical_terms = [t for c in clusters for t in c.get("terms", [])[:2]]
        critical_hits = {t: self._cluster_term_hits_dict(items, t) for t in flat_critical_terms}

        need_gap = len(cluster_gap) > 0

        return {
            "need_gap": need_gap,
            "clusters": cluster_hits,
            "cluster_gap": cluster_gap,
            "critical_hits": critical_hits,
            "general_hits": {},
            "critical_met": len(cluster_hits) - len(cluster_gap),
            "general_met": len(cluster_hits) - len(cluster_gap),
        }

    @staticmethod
    def _cluster_term_hits_dict(items: list[dict], term: str) -> int:
        if not term:
            return 0
        cnt = 0
        t = term.lower()
        for it in items or []:
            text = f"{it.get('title','')} {it.get('summary','')} {it.get('description','')} {it.get('article_body','')[:1500]}".lower()
            if t in text:
                cnt += 1
        return cnt

    def _build_gap_queries_for_cluster(self, customer: dict, cluster_name: str, cluster_data: dict) -> list[str]:
        queries = []
        terms = [t for t in (cluster_data.get("terms", []) or []) if isinstance(t, str) and t.strip()]
        # prioritize terms with low hit ratio
        for t in terms:
            if t not in queries:
                queries.append(t)
        if len(terms) == 0:
            queries.append(cluster_name)

        return queries[:10]

    def _build_gap_queries(self, customer: dict, coverage: dict) -> list[str]:
        # compatibility wrapper: keep previous aggregate behavior for any caller that may rely on it
        queries = []
        for c_name, c_data in (coverage.get("cluster_gap", {}) or {}).items():
            queries.extend(self._build_gap_queries_for_cluster(customer, c_name, c_data))

        if not queries:
            for k, v in (coverage.get("critical_hits", {}) or {}).items():
                if v < 1 and k not in queries:
                    queries.append(k)
        return queries[:10]

    def _apply_country_floor(self, selected: list[dict], fallback_pool: list[dict], country_floor: dict) -> list[dict]:
        if not country_floor:
            return selected
        out = list(selected or [])
        used = set([(it.get("url") or it.get("source_url") or "").strip().lower() for it in out])

        def count_country(c):
            return sum(1 for x in out if x.get("country", "GLOBAL") == c)

        # prefer already-ranked fallback order
        for country, need_n in country_floor.items():
            while count_country(country) < int(need_n):
                cand = None
                for it in fallback_pool or []:
                    u = (it.get("url") or it.get("source_url") or "").strip().lower()
                    if not u or u in used:
                        continue
                    if it.get("country", "GLOBAL") != country:
                        continue
                    cand = it
                    break
                if cand is None:
                    break
                out.append(cand)
                used.add((cand.get("url") or cand.get("source_url") or "").strip().lower())
        return out

    def _fill_minimum_articles(self, selected: list[dict], fallback_pool: list[dict], min_articles: int) -> list[dict]:
        out = list(selected or [])
        if len(out) >= int(min_articles):
            return out
        used = set([(it.get("url") or it.get("source_url") or "").strip().lower() for it in out])
        for it in fallback_pool or []:
            if len(out) >= int(min_articles):
                break
            u = (it.get("url") or it.get("source_url") or "").strip().lower()
            if not u or u in used:
                continue
            out.append(it)
            used.add(u)
        return out

    def _build_country_gap_queries(self, customer: dict, selected: list[dict], country_gap: int) -> list[str]:
        existing = set([it.get("country", "GLOBAL") for it in (selected or []) if it.get("country")])
        order = self.config.get("news", {}).get("country_order", ["KR", "US", "CN", "TW", "GLOBAL"])
        missing = [c for c in order if c not in existing]

        country_hint = {
            "KR": "Korea AI semiconductor data center cloud",
            "US": "US AI semiconductor data center cloud",
            "CN": "China AI semiconductor policy supply chain",
            "TW": "Taiwan AI semiconductor foundry packaging",
            "GLOBAL": "global AI infrastructure semiconductor supply chain",
        }

        out = []
        for c in missing[: max(1, int(country_gap))]:
            q = country_hint.get(c)
            if q:
                out.append(q)
        return out

    def _build_ko_en_query_pairs(self, queries: list[str]) -> list[str]:
        out = []
        for q in queries or []:
            qq = (q or "").strip()
            if not qq:
                continue
            out.append(qq)
            out.append(f"{qq} AI semiconductor")
        # dedup preserve order
        seen = set()
        deduped = []
        for q in out:
            k = q.lower().strip()
            if not k or k in seen:
                continue
            seen.add(k)
            deduped.append(q)
        return deduped

    def _rebalance_domain_bias(self, items: list[dict], max_share: float = 0.3) -> list[dict]:
        if not items:
            return items
        total = len(items)
        cap = max(1, int(total * max_share))
        kept = []
        dom_cnt = {}
        for it in items:
            u = (it.get("url") or it.get("source_url") or "").strip().lower()
            d = self._normalize_domain(urlparse(u).netloc or "") if u else ""
            d = d or "unknown"
            if dom_cnt.get(d, 0) >= cap:
                continue
            dom_cnt[d] = dom_cnt.get(d, 0) + 1
            kept.append(it)
        return kept

    def _normalize_domain(self, domain: str) -> str:
        d = (domain or "").strip().lower()
        if d.startswith("http://") or d.startswith("https://"):
            d = (urlparse(d).netloc or "").lower()
        if d.startswith("www."):
            d = d[4:]
        return d

    def _load_forbidden_domains(self) -> set[str]:
        p = os.path.join(self.base_dir, "..", "newsletter_agent.json")
        domains = set()
        try:
            cfg = json.loads(Path(p).read_text(encoding="utf-8")) if os.path.exists(p) else {}
            arr = (
                cfg.get("consistencyValidation", {})
                .get("linkValidation", {})
                .get("forbidDomains", [])
            )
            for d in arr or []:
                n = self._normalize_domain(str(d))
                if n:
                    domains.add(n)
        except Exception:
            pass
        return domains

    def _is_forbidden_url(self, url: str) -> bool:
        u = (url or "").strip().lower()
        if not u:
            return False
        host = self._normalize_domain(urlparse(u).netloc or "")
        for d in self.forbidden_domains:
            if host == d or host.endswith(f".{d}") or d in u:
                return True
        return False

    def _validate_and_filter_sources(self):
        cats = self.sources.get("categories", {}) if isinstance(self.sources, dict) else {}
        removed = 0
        for _, cfg in cats.items():
            rss = cfg.get("rss", []) or []
            kept = []
            for u in rss:
                if self._is_forbidden_url(u):
                    removed += 1
                    continue
                kept.append(u)
            cfg["rss"] = kept
        if removed:
            self.log.warning(f"Filtered forbidden RSS sources by policy: removed={removed}")

    def _validate_pre_send(
        self,
        customer_id: str,
        issue_date: str,
        processed_scan: dict[str, list[dict]],
        html: str,
        template_path: str,
        prior_errors: Optional[list[str]] = None,
    ) -> tuple[bool, list[str], list[str]]:
        hard_errs = []
        soft_warns = []

        for e in list(prior_errors or []):
            if str(e).startswith("global_scan<"):
                soft_warns.append(str(e))
            else:
                hard_errs.append(str(e))

        total_scan = sum(len(v) for v in processed_scan.values())
        if total_scan < 8:
            soft_warns.append(f"soft_min_articles_not_met:{total_scan}<8")

        active_countries = [c for c, items in processed_scan.items() if items]
        if len(active_countries) < 2:
            soft_warns.append(f"soft_min_country_sections_not_met:{len(active_countries)}<2")

        if "CLUE_TEMPLATE_OFFICIAL.html" not in (template_path or ""):
            hard_errs.append("template_path_not_official")

        unresolved = [
            "{{ARTICLE_PRACTICAL_IMPLICATION}}",
            "{{NEEDS_HASHTAGS}}",
            "{{SERIAL_NUMBER}}",
        ]
        for token in unresolved:
            if token in html:
                hard_errs.append(f"unresolved_placeholder:{token}")

        # 금지 문구는 하드 차단 대신 소프트 경고로 기록 (생성 프롬프트로 예방)
        banned_phrases = [
            "기사에는",
            "기사 본문에서는",
            "기사에서는",
            "TechCrunch 기사에 따르면",
            "TechCrunch 보도는",
            "본문은",
            "제시합니다",
            "평가합니다",
            "로 볼 수 있습니다",
            "실무에서는",
            "SK hynix에서는",
        ]
        html_low = (html or "").lower()
        for phrase in banned_phrases:
            if phrase.lower() in html_low:
                soft_warns.append(f"forbidden_phrase_detected:{phrase}")

        # 기사 요약/실무시사점 분리 품질 체크
        for c, items in processed_scan.items():
            for idx, it in enumerate(items, start=1):
                desc = " ".join((it.get("description") or "").split())
                practical = " ".join((it.get("practical_implication") or "").split())
                if not practical:
                    hard_errs.append(f"missing_practical_implication:{c}:{idx}")
                    continue
                if desc and (desc in practical or practical in desc):
                    hard_errs.append(f"practical_too_similar_to_summary:{c}:{idx}")

        urls = []
        for items in processed_scan.values():
            for it in items:
                u = (it.get("url") or "").strip()
                if u:
                    urls.append(u)

        lower_urls = [u.lower() for u in urls]
        forbidden_hits = [u for u in lower_urls if self._is_forbidden_url(u)]
        if forbidden_hits:
            hard_errs.append(f"forbidden_domain_detected:{len(forbidden_hits)}")

        unique_count = len(set(lower_urls))
        if unique_count != len(lower_urls):
            soft_warns.append(f"duplicate_article_url_detected:{len(lower_urls)-unique_count}")

        valid_scheme = [u for u in lower_urls if u.startswith("http://") or u.startswith("https://")]
        non_forbidden = [u for u in valid_scheme if not self._is_forbidden_url(u)]
        pass_rate = (len(non_forbidden) / len(lower_urls)) if lower_urls else 0.0
        if pass_rate < 0.7:
            soft_warns.append(f"soft_link_pass_rate_below_threshold:{pass_rate:.2f}<0.70")

        html_low = (html or "").lower()
        if any(d in html_low for d in self.forbidden_domains):
            hard_errs.append("forbidden_domain_detected_in_html")

        # 히스토리/타 고객 대비 URL 중복 경고는 비활성화

        return (len(hard_errs) == 0, hard_errs, soft_warns)

    def _proof_path(self) -> Path:
        return Path(self.base_dir) / "data" / "customer_url_proof.json"

    def _load_url_proof(self) -> dict:
        p = self._proof_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _record_customer_urls(self, customer_id: str, issue_date: str, urls: list[str]):
        if not urls:
            return
        p = self._proof_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self._load_url_proof()
        bucket = data.get(issue_date, {})
        bucket[customer_id] = sorted(set(urls))
        data[issue_date] = bucket
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _cross_customer_overlap_ratio(self, customer_id: str, issue_date: str, urls: list[str]) -> tuple[float, Optional[str]]:
        current = set(urls)
        if not current:
            return (0.0, None)

        data = self._load_url_proof()
        same_day = data.get(issue_date, {}) if isinstance(data, dict) else {}
        max_ratio = 0.0
        max_other = None

        for other_id, other_urls in same_day.items():
            if other_id == customer_id:
                continue
            other_set = set(other_urls or [])
            if not other_set:
                continue
            inter = len(current & other_set)
            denom = max(1, min(len(current), len(other_set)))
            ratio = inter / denom
            if ratio > max_ratio:
                max_ratio = ratio
                max_other = other_id

        return (max_ratio, max_other)

    def _append_jsonl(self, path: str, payload: dict):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _append_run_log(self, customer_id: str, result: dict):
        payload = {
            "timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "type": "agent_run",
            "customerId": customer_id,
            "status": result.get("status"),
            "gateOk": result.get("gate_ok"),
            "gateErrors": result.get("gate_errors", []),
            "totalScan": result.get("total_scan", 0),
            "delivery": result.get("delivery", {}),
            "templatePath": result.get("template_path", ""),
            "templateSha256": result.get("template_sha256", ""),
        }
        self._append_jsonl(os.path.join(self.base_dir, "..", "newsletter_runs.jsonl"), payload)

    def _append_audit_log(self, customer_id: str, result: dict):
        payload = {
            "at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "action": "send_attempt",
            "customerId": customer_id,
            "status": result.get("status"),
            "gateOk": result.get("gate_ok"),
            "gateErrors": result.get("gate_errors", []),
            "templatePath": result.get("template_path", ""),
            "templateSha256": result.get("template_sha256", ""),
        }
        self._append_jsonl(os.path.join(self.base_dir, "..", "newsletter_audit.jsonl"), payload)

    def _is_publish_enabled(self) -> bool:
        cfg_path = os.path.join(self.base_dir, "..", "newsletter_agent.json")
        try:
            if os.path.exists(cfg_path):
                cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
                return bool(cfg.get("publishSwitch", {}).get("enabled", True))
        except Exception:
            pass
        return True

    def _select_for_customer(self, customer: dict, pool: list[dict]) -> list[dict]:
        prefs = customer.get("preferences", {})
        categories = set(prefs.get("categories", []))
        known_categories = set((self.sources.get("categories") or {}).keys()) | {"GENERAL"}
        # 고객 토픽이 수집 카테고리 체계와 다르면 카테고리 필터 비활성화
        if categories and not (categories & known_categories):
            categories = set()

        clusters = self._extract_need_clusters(customer)
        cluster_terms = []
        for c in clusters:
            cluster_terms.extend([x for x in c.get("terms", []) if isinstance(x, str) and x.strip()])
        keywords = [k.lower() for k in prefs.get("keywords", [])]
        keywords = sorted(set([k.lower() for k in keywords + [x.lower() for x in cluster_terms]]))
        watch_companies = [c.lower() for c in prefs.get("watch_companies", []) if isinstance(c, str)]
        cluster_specs = []
        for c in clusters:
            c_name = str(c.get("name", "cluster"))
            c_terms = [x.lower() for x in c.get("terms", []) if isinstance(x, str) and x.strip()]
            c_min = max(1, int(c.get("minArticles", 0) or 1))
            c_weight = float(c.get("weight", 1.0) or 1.0)
            cluster_specs.append((c_name, c_terms, c_min, c_weight))

        scoped = []
        for a in pool:
            if categories and a.get("category") not in categories:
                continue
            # 카테고리 적합도 최소 게이트
            if a.get("_category_score", 0) < 0:
                continue
            # 니즈와 무관한 글로벌/라이프스타일성 기사 제외
            if not self._is_customer_relevant(a, keywords):
                continue
            scoped.append(a)

        for a in scoped:
            text = f"{a.get('title','')} {a.get('summary','')}".lower()
            keyword_score = sum(text.count(k) for k in keywords)
            company_hits = sum(1 for c in watch_companies if c and c in text)
            a["_customer_score"] = keyword_score + (company_hits * 3)
            cluster_scores = {}
            for c_name, c_terms, _, c_weight in cluster_specs:
                s = 0
                for t in c_terms:
                    s += text.count(t)
                cluster_scores[c_name] = s * c_weight
            a["_cluster_scores"] = cluster_scores

        # score desc + recency
        scoped.sort(
            key=lambda x: (x.get("_customer_score", 0), x.get("_category_score", 0), x.get("published_at", "")),
            reverse=True,
        )

        max_pool = 80
        target_scan = self.config["news"]["global_scan"].get("target_per_newsletter", 10)

        # 1차: 클러스터 최소치 보장 슬롯링 선점
        selected = []
        used = set()
        if cluster_specs:
            # buckets by cluster
            buckets = []
            for c_name, _, c_min, _ in cluster_specs:
                bucket = [a for a in scoped if a.get("_cluster_scores", {}).get(c_name, 0) > 0 and id(a) not in used]
                bucket = self._dedupe_similar_items(bucket, text_key="title")
                buckets.append((c_name, c_min, bucket))

            for c_name, c_min, bucket in buckets:
                while len([1 for x in selected if x.get("_cluster_scores", {}).get(c_name, 0) > 0]) < c_min:
                    if not bucket:
                        break
                    a = bucket.pop(0)
                    uid = id(a)
                    if uid in used:
                        continue
                    selected.append(a)
                    used.add(uid)
                if len(selected) >= max_pool:
                    break

            if len(selected) < target_scan:
                for a in scoped:
                    uid = id(a)
                    if uid in used:
                        continue
                    selected.append(a)
                    used.add(uid)
                    if len(selected) >= max_pool:
                        break
        else:
            selected = list(scoped)

        if len(selected) >= target_scan:
            return self._dedupe_similar_items(selected[:max_pool], text_key="title")

        # fallback: 부족분은 카테고리 적합도가 높은 기사로만 보강
        for a in scoped:
            uid = id(a)
            if uid in used:
                continue
            if a.get("_category_score", 0) < 2:
                continue
            selected.append(a)
            used.add(uid)
            if len(selected) >= max_pool:
                break

        return self._dedupe_similar_items(selected, text_key="title")

    def _group_by_country(self, items: list[dict]) -> dict[str, list[dict]]:
        out = {c: [] for c in self.config["news"].get("country_order", ["KR", "US", "CN", "TW", "GLOBAL"])}
        for a in items:
            c = a.get("country", "GLOBAL")
            if c not in out:
                c = "GLOBAL"
            out[c].append(a)
        return out

    def _select_research_for_customer(self, customer: dict, research_items: list[dict]) -> list[dict]:
        prefs = customer.get("preferences", {})
        keywords = [k.lower() for k in prefs.get("keywords", []) if isinstance(k, str)]

        for r in research_items:
            text = f"{r.get('title','')} {r.get('summary','')} {r.get('source','')}".lower()
            keyword_score = sum(text.count(k) for k in keywords)
            source_score = int(r.get("_score", 0) or 0)
            r["_customer_score"] = keyword_score
            r["_rank_score"] = keyword_score * 3 + source_score

        ranked = sorted(
            research_items,
            key=lambda x: (x.get("_rank_score", 0), x.get("published_at", "")),
            reverse=True,
        )

        max_research = self.config["news"]["research_insight"].get("max_items", 5)
        min_research = self.config["news"]["research_insight"].get("target_per_newsletter", 2)
        pick_n = max(max_research, min_research)

        # source_type 기반 버킷 믹스 (설정 기반)
        mix_cfg = self.sources.get("research", {}).get("target_mix", {})
        type_map = {
            "academic": {"academic"},
            "consulting": {"research_firm"},
            "institute": {"university", "korea_research", "institute"},
        }

        selected = []
        used_urls = set()

        def pick_from_types(type_keys: set[str], n: int):
            nonlocal selected
            for r in ranked:
                if n <= 0:
                    break
                u = (r.get("url") or "").lower().strip()
                if not u or u in used_urls:
                    continue
                if r.get("source_type") not in type_keys:
                    continue
                if r.get("_customer_score", 0) <= 0:
                    continue
                selected.append(r)
                used_urls.add(u)
                n -= 1

        # 1차: 버킷별 최소 quota 확보
        for bucket, n in mix_cfg.items():
            try:
                quota = int(n)
            except Exception:
                quota = 0
            if quota <= 0:
                continue
            pick_from_types(type_map.get(bucket, set()), quota)

        # 2차: 남는 슬롯은 전체 점수순 보강
        if len(selected) < pick_n:
            for r in ranked:
                u = (r.get("url") or "").lower().strip()
                if not u or u in used_urls:
                    continue
                if r.get("_customer_score", 0) <= 0 and len(selected) >= min_research:
                    continue
                selected.append(r)
                used_urls.add(u)
                if len(selected) >= pick_n * 2:
                    break

        selected = self._dedupe_similar_items(selected, text_key="title")
        return selected[:pick_n]

    def _infer_category(self, article: dict) -> tuple[str, int]:
        text = f"{article.get('title','')} {article.get('summary','')}".lower()
        categories = self.sources.get("categories", {})
        best_cat = article.get("source_category", "GENERAL")
        best_score = -1

        for cat, cfg in categories.items():
            rules = self.category_rules.get(cat, {}) if isinstance(self.category_rules, dict) else {}

            matched = set()
            for kw in cfg.get("keywords", []):
                k = str(kw).lower()
                if k and k in text:
                    matched.add(k)
            score = len(matched)

            # include 가점
            for kw in rules.get("include_keywords", []):
                k = str(kw).lower()
                if k and k in text:
                    score += 1

            # exclude 감점
            for kw in rules.get("exclude_keywords", []):
                k = str(kw).lower()
                if k and k in text:
                    score -= 1

            # hard exclude는 최소화 적용: 즉시 제외가 아니라 큰 감점
            for kw in rules.get("hard_exclude_keywords", []):
                k = str(kw).lower()
                if k and k in text:
                    score -= 3

            # 기존 소스 카테고리에 약한 prior
            if cat == article.get("source_category"):
                score += 1

            if score > best_score:
                best_cat = cat
                best_score = score

        # 최소 근거 점수 미달 시 일반 기사로 분류
        if best_score < 2:
            return "GENERAL", max(best_score, 0)
        return best_cat, max(best_score, 0)

    def _dedupe_similar_items(self, items: list[dict], text_key: str = "title") -> list[dict]:
        out = []
        for it in items:
            txt = (it.get(text_key) or "").strip()
            if not txt:
                continue
            if any(self._is_similar_title(txt, (x.get(text_key) or "")) for x in out):
                continue
            out.append(it)
        return out

    def _dedupe_by_title_strict(self, items: list[dict]) -> list[dict]:
        out = []
        for it in items:
            title = (it.get("title") or "").strip()
            if not title:
                continue

            # 1) 기존 결과와 제목 유사도 비교(강화)
            if any(self._is_similar_title(title, (x.get("title") or ""), jaccard_threshold=0.38) for x in out):
                continue

            # 2) 요약 첫 문장 기준 중복 제거(같은 이슈의 제목 변형 대응)
            s = " ".join((it.get("summary") or "").split())
            s = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", s.lower())).strip()
            first_clause = s[:120]
            if first_clause and any(first_clause == (x.get("_dedupe_clause") or "") for x in out):
                continue

            it["_dedupe_clause"] = first_clause
            out.append(it)

        for x in out:
            x.pop("_dedupe_clause", None)
        return out

    def _title_summary_consistency_score(self, title: str, summary: str) -> float:
        t = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", (title or "").lower())).strip()
        s = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", (summary or "").lower())).strip()
        if not t or not s:
            return 0.0
        t_toks = {x for x in t.split() if len(x) >= 2}
        s_toks = {x for x in s.split() if len(x) >= 2}
        if not t_toks or not s_toks:
            return 0.0
        overlap = len(t_toks & s_toks) / max(1, len(t_toks))
        return overlap

    def _extract_title_anchors(self, title: str) -> list[str]:
        t = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", (title or "").lower())).strip()
        toks = [x for x in t.split() if len(x) >= 2]
        stop = {"관련", "통해", "대한", "위한", "대한", "및", "에서", "으로", "까지", "발표", "출시"}
        anchors = []
        for tok in toks:
            if tok in stop:
                continue
            anchors.append(tok)
            if len(anchors) >= 6:
                break
        return anchors

    def _is_title_summary_fully_aligned(self, title: str, summary: str) -> bool:
        score = self._title_summary_consistency_score(title, summary)
        s = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", (summary or "").lower())).strip()
        anchors = self._extract_title_anchors(title)
        if not anchors:
            return False
        # strict gate: high overlap + all core anchors present in summary text
        anchors_ok = all(a in s for a in anchors[:4])
        return score >= 0.60 and anchors_ok

    def _enforce_title_summary_consistency(self, processed_scan: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """제목-요약 하드코딩 정합성 게이트 비활성화.

        정책상 제목/요약은 동일 URL 본문 기반 + LLM 본문 근거 판정을 우선하므로,
        이 단계에서는 비어있는 항목만 최소 필터링한다.
        """
        out = {}
        for c, items in (processed_scan or {}).items():
            kept = []
            for it in items:
                title = (it.get("title_ko") or it.get("title") or "").strip()
                desc = (it.get("description") or "").strip()
                if not title or not desc:
                    continue
                kept.append(it)
            out[c] = kept
        return out

    def _dedupe_semantic(self, items: list[dict]) -> list[dict]:
        """Policy-linked semantic dedup execution.
        Keep one representative per near-duplicate event cluster.
        """
        out = []
        for it in items:
            title = (it.get("title") or "").strip()
            summary = " ".join((it.get("summary") or "").split())
            url = (it.get("url") or "").strip().lower()
            if not title:
                continue

            t_norm = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", title.lower())).strip()
            s_norm = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", summary.lower())).strip()
            sig = f"{t_norm[:90]}|{s_norm[:140]}"

            duplicate_idx = None
            for idx, kept in enumerate(out):
                k_title = (kept.get("title") or "")
                k_sum = " ".join((kept.get("summary") or "").split())
                k_url = (kept.get("url") or "").strip().lower()

                same_url = bool(url and k_url and url == k_url)
                same_host = False
                try:
                    same_host = (urlparse(url).netloc == urlparse(k_url).netloc)
                except Exception:
                    pass

                close_title = self._is_similar_title(title, k_title, jaccard_threshold=0.34)
                close_summary = self._is_similar_title(summary, k_sum, jaccard_threshold=0.40)
                close_sig = self._is_similar_title(sig, f"{k_title} {k_sum}", jaccard_threshold=0.42)

                if same_url or close_title or (same_host and close_summary) or close_sig:
                    duplicate_idx = idx
                    break

            if duplicate_idx is None:
                out.append(it)
                continue

            kept = out[duplicate_idx]
            new_cons = self._title_summary_consistency_score(title, summary)
            old_cons = self._title_summary_consistency_score(kept.get("title", ""), kept.get("summary", ""))
            new_info = len((summary or "").split()) + len((title or "").split()) * 2
            old_info = len((kept.get("summary") or "").split()) + len((kept.get("title") or "").split()) * 2

            # representative priority: title-summary consistency > info density
            if (new_cons > old_cons + 0.03) or (abs(new_cons - old_cons) <= 0.03 and new_info > old_info):
                out[duplicate_idx] = it

        return out

    @staticmethod
    def _is_customer_relevant(article: dict, keywords: list[str]) -> bool:
        text = f"{article.get('title','')} {article.get('summary','')}".lower()

        # 최소 제외 룰만 적용(볼륨 유지)
        hard_excludes = [
            "disney", "hulu", "coupon", "promo", "box office",
            "celebrity", "bet", "gambling", "cash apples",
        ]
        if any(x in text for x in hard_excludes):
            return False

        def has_term(term: str) -> bool:
            t = (term or "").strip().lower()
            if not t:
                return False
            if re.search(r"[a-z0-9]", t):
                return re.search(rf"\b{re.escape(t)}\b", text) is not None
            return t in text

        # 고객 키워드 또는 클러스터 기반 키워드 매칭은 통과,
        # keywords가 비어도 앵커 최소 기준을 둬 라이프클러스터 유입은 제한
        if any(has_term(k) for k in keywords):
            return True

        anchor_terms = [
            "ai", "agent", "llm", "rag", "smart factory", "semiconductor", "digital twin", "automation",
            "travel", "camping", "festival", "concert", "restaurant",
        ]
        return any(has_term(k) for k in anchor_terms)

    @staticmethod
    def _is_similar_title(a: str, b: str, jaccard_threshold: float = 0.45) -> bool:
        if not a or not b:
            return False
        a_norm = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", a.lower())).strip()
        b_norm = re.sub(r"\s+", " ", re.sub(r"[^\w가-힣 ]", " ", b.lower())).strip()
        if not a_norm or not b_norm:
            return False
        if a_norm == b_norm:
            return True
        if a_norm in b_norm or b_norm in a_norm:
            return True
        def norm_tok(t: str) -> str:
            t = t.strip()
            for p in ("은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "로", "도"):
                if t.endswith(p) and len(t) >= 3:
                    t = t[: -len(p)]
                    break
            return t

        a_toks = {norm_tok(t) for t in a_norm.split() if len(t) >= 2}
        b_toks = {norm_tok(t) for t in b_norm.split() if len(t) >= 2}
        if not a_toks or not b_toks:
            return False
        jac = len(a_toks & b_toks) / len(a_toks | b_toks)
        if jac >= jaccard_threshold:
            return True

        company_terms = {"openai", "google", "anthropic", "claude", "gemini", "microsoft", "meta", "tencent", "xiaomi"}
        action_terms = {"소송", "고소", "lawsuit", "sue", "release", "출시", "launch", "update", "업데이트"}
        if (a_toks & b_toks & company_terms) and (a_toks & b_toks & action_terms):
            return True

        # 회사 + 동일 기관(예: 국방부) + 양쪽 모두 액션 단어 보유 시 유사 기사로 간주
        gov_terms = {"국방부", "department", "defense", "정부", "government"}
        a_has_action = bool(a_toks & action_terms)
        b_has_action = bool(b_toks & action_terms)
        if (a_toks & b_toks & company_terms) and (a_toks & b_toks & gov_terms) and a_has_action and b_has_action:
            return True

        # Anthropic-국방부 소송 축의 유사 헤드라인 통합
        if ("anthropic" in a_toks and "anthropic" in b_toks) and (("국방부" in a_toks and "국방부" in b_toks) or ("defense" in a_toks and "defense" in b_toks)):
            return True

        return False

    def _build_customer_queries(self, customer: dict) -> list[str]:
        prefs = customer.get("preferences", {})

        # 우선순위 1: 고객별 명시 쿼리
        explicit = [q.strip() for q in prefs.get("search_queries", []) if isinstance(q, str) and q.strip()]

        companies = [c.strip() for c in prefs.get("watch_companies", []) if isinstance(c, str) and c.strip()]
        focus_topics = [t.strip() for t in prefs.get("focus_topics", []) if isinstance(t, str) and t.strip()]

        if not companies:
            seed_keywords = [k for k in prefs.get("keywords", []) if isinstance(k, str)][:8]
            focus_topics = focus_topics or seed_keywords[:4]
            companies = seed_keywords[:4]

        queries = explicit[:]

        if companies or focus_topics:
            focus_topics = focus_topics or ["AI", "enterprise adoption"]
            companies = companies[:8]
            focus_topics = focus_topics[:5]
            for c in companies:
                for t in focus_topics:
                    queries.append(f"{c} {t}")

        # 모든 고객 공통 보강 (국가/산업 균형 확보)
        queries.extend([
            "China AI policy technology development",
            "Chinese LLM startup DeepSeek Moonshot",
            "US enterprise AI agent adoption",
            "Korea AI semiconductor industry trend",
            "Taiwan AI supply chain TSMC",
            "global physical AI robotics enterprise deployment",
            "AI data center investment hyperscaler",
            "AI governance compliance enterprise",
        ])

        uniq = []
        seen = set()
        for q in queries:
            k = q.lower().strip()
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(q)
        return uniq[:40]

    @staticmethod
    def _build_needs_hashtags(customer: dict, max_tags: int = 5) -> str:
        prefs = customer.get("preferences", {}) if isinstance(customer, dict) else {}
        keywords = [k for k in prefs.get("keywords", []) if isinstance(k, str) and k.strip()]
        tags = []
        seen = set()
        for kw in keywords:
            t = kw.strip().replace("#", "")
            if not t:
                continue
            t = t.replace(" ", "")
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            tags.append(f"#{t}")
            if len(tags) >= max_tags:
                break
        return " ".join(tags)

    def _load_customers(self) -> list[dict]:
        preferred = os.path.join(self.base_dir, "..", "newsletter_customers.json")
        legacy = os.path.join(self.base_dir, "data", "customers.json")

        raw = {}
        source = None
        if os.path.exists(preferred):
            source = preferred
            with open(preferred, "r", encoding="utf-8") as f:
                raw = json.load(f)
        elif os.path.exists(legacy):
            source = legacy
            with open(legacy, "r", encoding="utf-8") as f:
                raw = json.load(f)

        if isinstance(raw, dict) and isinstance(raw.get("customers"), list):
            customers = [self._normalize_customer_record(c) for c in raw.get("customers", [])]
            customers = [c for c in customers if c.get("enabled", True)]
            self.log.info(f"Loaded customers from {source}: {len(customers)} active")
            return customers

        if isinstance(raw, list):
            self.log.info(f"Loaded legacy customers from {source}: {len(raw)}")
            return raw

        self.log.warning("No customer source found; using empty list")
        return []

    @staticmethod
    def _normalize_customer_record(c: dict) -> dict:
        profile = c.get("needsProfile", {}) if isinstance(c, dict) else {}
        normalized = {
            "customer_id": c.get("customerId") or c.get("customer_id") or "default",
            "name": c.get("name", ""),
            "email": c.get("email", ""),
            "serial_number": c.get("serialNumber") or c.get("serial_number") or "",
            "enabled": c.get("enabled", True),
            "preferences": {
                "categories": profile.get("topics", []),
                "keywords": profile.get("keywords", []),
                "excludes": profile.get("excludes", []),
                "search_queries": profile.get("searchQueries", []),
                "needClusters": profile.get("needClusters", []),
            },
        }

        # 기존 쿼리 확장 로직과의 호환
        if profile.get("prioritySignals"):
            normalized["preferences"]["focus_topics"] = profile.get("prioritySignals", [])

        return normalized

    def _get_customer(self, customer_id: str) -> dict:
        for c in self.customers:
            if c.get("customer_id") == customer_id:
                return c
        return self.customers[0] if self.customers else {"customer_id": customer_id, "preferences": {}}

    def _load_yaml(self, rel_path: str):
        with open(os.path.join(self.base_dir, rel_path), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_json(self, rel_path: str):
        p = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(p):
            return []
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_env_vars(self, node):
        if isinstance(node, dict):
            return {k: self._resolve_env_vars(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._resolve_env_vars(v) for v in node]
        if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
            key = node[2:-1]
            return os.getenv(key, "")
        return node

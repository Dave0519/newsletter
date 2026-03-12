from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

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

        # PHASE 1: collect once + customer-focused query expansion
        article_pool = self.news_collector.collect_all(per_category_limit=30)
        custom_queries = self._build_customer_queries(customer)
        if custom_queries:
            article_pool.extend(self.news_collector.collect_custom_queries(custom_queries, category="AI_TECH", limit=50))
            article_pool = self.news_collector._dedup_pool(article_pool)

        research_cfg = self.config["news"].get("research_insight", {})
        research_enabled = bool(research_cfg.get("enabled", True))
        research_target = max(
            research_cfg.get("max_items", 5),
            research_cfg.get("target_per_newsletter", 2),
        )
        raw_research = []
        if research_enabled and research_target > 0:
            # 리서치 개인화 필터링을 위해 기본치보다 넉넉하게 수집
            raw_research = self.research_collector.collect(target_count=max(research_target * 4, 12))

        # PHASE 2: content-based tagging
        for a in article_pool:
            a["country"] = self.country_tagger.tag(a.get("title", ""), a.get("summary", ""))
            cat, score = self._infer_category(a)
            a["category"] = cat
            a["_category_score"] = score

        # PHASE 3: customer personalization
        selected = self._select_for_customer(customer, article_pool)
        selected = self._dedupe_by_title_strict(selected)
        selected = self._dedupe_semantic(selected)

        # shortage mitigation: 기사 부족 시 source/queries 확장 재수집
        target_scan = int(self.config["news"]["global_scan"].get("target_per_newsletter", 10))
        if len(selected) < target_scan:
            boost_queries = custom_queries[:]
            boost_queries.extend([
                "China AI policy technology development 2026",
                "Chinese LLM startup DeepSeek Moonshot 2026",
                "Reuters AI enterprise deployment 2026",
                "Nikkei Asia AI semiconductor strategy 2026",
                "Korea JoongAng Daily AI industry 2026",
                "MIT Technology Review AI enterprise 2026",
                "The Verge AI model update 2026",
                "Bloomberg AI enterprise rollout 2026",
                "Financial Times AI regulation enterprise 2026",
                "WSJ AI chip supply chain 2026",
                "OpenAI Anthropic Google enterprise agent deployment 2026",
                "NVIDIA AMD Intel AI data center roadmap 2026",
                "Samsung SK hynix HBM AI server demand 2026",
                "TSMC advanced packaging CoWoS AI demand 2026",
                "robotics physical AI manufacturing deployment 2026",
                "industrial AI agent workflow automation case study 2026",
                "Korea semiconductor AI policy 2026",
                "Taiwan AI semiconductor ecosystem 2026",
                "US China semiconductor export control AI 2026",
                "generative AI enterprise security governance 2026",
            ])
            # 중복 제거
            dedup_q = []
            seen_q = set()
            for q in boost_queries:
                k = (q or "").strip().lower()
                if not k or k in seen_q:
                    continue
                seen_q.add(k)
                dedup_q.append(q)

            refill = self.news_collector.collect_custom_queries(dedup_q, category="AI_TECH", limit=80)
            if refill:
                article_pool = self.news_collector._dedup_pool(article_pool + refill)
                for a in article_pool:
                    if "country" not in a:
                        a["country"] = self.country_tagger.tag(a.get("title", ""), a.get("summary", ""))
                    if "category" not in a:
                        cat, score = self._infer_category(a)
                        a["category"] = cat
                        a["_category_score"] = score
                selected = self._select_for_customer(customer, article_pool)
                selected = self._dedupe_by_title_strict(selected)
                selected = self._dedupe_semantic(selected)

        # 히스토리 기반 중복 제거는 비활성화 (테스트/재생성 시 동일 기사 허용)
        personalized_research = self._select_research_for_customer(customer, raw_research) if raw_research else []
        filtered_research = personalized_research if personalized_research else []

        # PHASE 4: regroup by country for existing template
        scan_by_country = self._group_by_country(selected)

        ok, errors = self.builder.validate(
            scan_by_country,
            filtered_research,
            min_scan=self.config["news"]["global_scan"]["target_per_newsletter"],
            min_research=research_cfg.get("target_per_newsletter", 0),
        )
        if not ok:
            self.log.warning(f"Validation failed: {errors}")

        max_per_country = self.config["news"]["global_scan"].get("max_per_country", 5)
        processed_scan = {
            c: self.processor.process_news_batch(items, lang="ko")[:max_per_country]
            for c, items in scan_by_country.items()
        }
        max_research = research_cfg.get("max_items", 0)
        processed_research = []
        if research_enabled and max_research > 0 and filtered_research:
            processed_research = self.processor.process_research_batch(filtered_research, lang="ko")[:max_research]

        # PHASE 5: build + deliver
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
            if delivery_result.get("email"):
                # 히스토리 기반 dedup 기록 비활성화
                pass
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
            "html": html,
        }
        self._append_run_log(customer_id, result)
        self._append_audit_log(customer_id, result)
        return result

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

        # 금지 문구 하드 게이트
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
                hard_errs.append(f"forbidden_phrase_detected:{phrase}")

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

        keywords = [k.lower() for k in prefs.get("keywords", [])]
        watch_companies = [c.lower() for c in prefs.get("watch_companies", []) if isinstance(c, str)]

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

        # score desc + recency
        scoped.sort(
            key=lambda x: (x.get("_customer_score", 0), x.get("_category_score", 0), x.get("published_at", "")),
            reverse=True,
        )

        max_pool = 80
        target_scan = self.config["news"]["global_scan"].get("target_per_newsletter", 10)

        # 1차: 니즈 키워드 매칭 기사 우선
        matched = [a for a in scoped if a.get("_customer_score", 0) > 0]
        matched = self._dedupe_similar_items(matched[:max_pool], text_key="title")
        if len(matched) >= target_scan:
            return matched

        # 2차: 부족분은 카테고리 적합도가 높은 기사로만 보강(니즈 하한 유지)
        merged = matched[:]
        for a in scoped:
            if a in merged:
                continue
            if a.get("_category_score", 0) < 2:
                continue
            merged.append(a)
            if len(merged) >= max_pool:
                break

        return self._dedupe_similar_items(merged, text_key="title")

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

            # representative selection priority: evidence_grounding > info density > recency > source quality(heuristic)
            kept = out[duplicate_idx]
            new_score = len((summary or "").split()) + len((title or "").split()) * 2
            old_score = len((kept.get("summary") or "").split()) + len((kept.get("title") or "").split()) * 2
            if new_score > old_score:
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

        # 고객 키워드 매칭 또는 공통 AI/제조 앵커 매칭이 있을 때만 통과
        if any(has_term(k) for k in keywords):
            return True

        anchor_terms = [
            "ai", "agent", "llm", "rag", "smart factory", "semiconductor", "digital twin", "automation"
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
            "China AI policy technology development 2026",
            "Chinese LLM startup DeepSeek Moonshot 2026",
            "US enterprise AI agent adoption 2026",
            "Korea AI semiconductor industry trend 2026",
            "Taiwan AI supply chain TSMC 2026",
            "global physical AI robotics enterprise deployment 2026",
            "AI data center investment hyperscaler 2026",
            "AI governance compliance enterprise 2026",
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

# 외부 크롤링 데이터 수신용 스펙 (clue_letter_dev2)

이 문서는 다른 팀/담당자에게 전달하기 위한 **clue_letter_dev2 데이터 수신 계약**입니다.

- 버전: `v1`
- 적용 브랜치: `clue_letter_dev2_sync`
- 기준 날짜: `YYYY-MM-DD`

---

## 1) 핵심 목표
- 외부에서 수집한 뉴스 원시/가공 데이터라도, 내부 파이프라인에서는 **공통 풀(total_news) 기준 계약**으로 정규화하여 수신.
- 운영 파이프라인은 `needs → total_news(shared pool) → daily_news(개인화) → writing` 고정.
- 기존 템플릿의 `{{#COUNTRIES}}` 구조를 유지하기 위해 `country_code` 기반 섹션 그룹핑이 쉬워야 함.

---

## 2) 수신 산출물(필수)
- `EXTERNAL_TOTAL_NEWS_BUNDLE_V1.jsonl` (또는 동등한 jsonl 파일)
- 각 라인 1개 뉴스 레코드

권장 파일명 예시:
- `clue_letter_ext_total_news_2026-03-19.jsonl`

---

## 3) 필수 스키마(총 3개)
- `standard_policy.json`
- `total_news.schema.json`
- `daily_news.schema.json`

현재 커밋 기준 경로:
- `clue_letter_dev2/standard_policy.json`
- `clue_letter_dev2/total_news.schema.json`
- `clue_letter_dev2/daily_news.schema.json`

`daily_news.schema.json`는 렌더링 결과 스키마이고, 실제 수신시점에는 `total_news` 중심 스키마를 우선 적용.

---

## 4) 반드시 지켜야 할 입력 필드 (총알 규칙)
수신 레코드(`total_news`)는 다음을 반드시 포함:
- `title` (`string`, 제목)
- `url` (`uri`, 원문 URL)
- `country_code` (`enum`: `KR`, `US`, `CN`, `TW`, `GLOBAL`)
- `matched_needs` (`array[string]`)
- `matched_need_ids` (`array[string]`)
- `matched_aliases` (`array[string]`)
- `query` (`string`)
- `source_type` (`enum`: `direct`, `google`, `rss`)
- `source` (`string`)
- `extraction_status` (`string`, 예: `success_full`)
- `collected_at` (`date-time`)

---

## 5) 금지/필터 규칙
- 금지 도메인: `news.google.com`, `digitimes.com`
- 중복 키(`url/title/signature`) 중복 제외
- 마케팅/프로모션성 제목/본문은 필터링
  - 예: `할인`, `세일`, `쿠폰`, `이벤트`, `특가`, `가격`, `구매`, 쇼핑 URL 패턴 등

---

## 6) 추천/권장 필드
- `need_category`, `summary`, `summary_snippet`
- `need_match_score`, `source_score`, `relevance_score`, `relevance_note`
- `origin_type`, `origin_detail`, `issue_angle_id`
- `published_at`, `recency_score`
- `normalized_url`, `normalized_title`, `semantic_signature` (중복 제거 보조)

---

## 7) 요청 샘플 (JSONL 1개)
```json
{
  "title": "AMD, Samsung deepen AI chip ties with HBM4 supply and foundry talks",
  "url": "https://some-news-site.example.com/article/123",
  "country_code": "GLOBAL",
  "matched_needs": ["삼성전자", "HBM"],
  "matched_need_ids": ["삼성동향", "HBM"],
  "matched_aliases": ["Samsung", "HBM", "HPC"],
  "query": "삼성동향|HBM 뉴스",
  "source_type": "google",
  "source": "external-crawler",
  "extraction_status": "success_full",
  "collected_at": "2026-03-19T14:12:19.999063+09:00",
  "need_category": "HBM",
  "summary": "...",
  "need_match_score": 3.1,
  "source_score": 0.8,
  "relevance_score": 3.4,
  "issue_angle_id": "삼성동향",
  "origin_type": "external",
  "origin_detail": "partner-crawler"
}
```

---

## 8) 수신 후 내부 처리(개요)
1. 수신 파일을 total_news 계약으로 검증.
2. `country_code` 값 정규화/기본값 보정 후 공통 풀 저장.
3. 사용자별 `needs` 매칭으로 `daily_news`를 선별.
4. HOT_TOPIC는 모든 사용자 공통 섹션으로 보장.
5. `daily_news`는 `HOT_TOPIC, KR, US, CN, TW, GLOBAL` 순으로 렌더링.

---

## 9) 팀 전달 텍스트(복붙용)
> 외부 수집 데이터는 `clue_letter_dev2` 수신 계약(v1)로 주세요.
> 필수 JSONL 필드는 `title,url,country_code,matched_needs,matched_need_ids,matched_aliases,query,source_type,source,extraction_status,collected_at` 입니다.
> `country_code`는 `KR/US/CN/TW/GLOBAL`만 허용, `news.google.com`/`digitimes.com`은 제외, 중복 URL/title/signature는 선제 제거입니다.
> 기준 스키마는 `standard_policy.json`, `total_news.schema.json`, `daily_news.schema.json`를 참고하세요.

# Openclaw 개인화 2단계 개선 문서

## 문서 목적

이 문서는 `openclaw`의 다음 품질 개선 단계에서 무엇을 바꿨는지 설명합니다.

이번 단계의 범위는 아래 세 가지입니다.

- 개인화 상류 풀을 고객 query 기반으로 확장
- SK hynix 공통 hot topic을 동적으로 주입
- 최종 selection 직전에 overlap-aware rerank 적용

이 문서는 단순 사용법이 아니라, 왜 이 변경이 필요했고 무엇이 실제로 달라졌는지 정리하는 문서입니다.

---

## 기존 문제점

### 1. 상류 후보 풀이 너무 정적이었음

기존 구조에서는 고객별 `searchQueries`와 `needClusters`가 존재해도, 실제 Stage A shared pool은 정적 `sources.yaml` 기반으로 먼저 만들어졌습니다.

즉 고객별 의도는 아래 시점에 너무 늦게 반영됐습니다.

- Stage A: 정적 공용 pool 생성
- Stage B/C/D: 공용 pool 기반 shortlist, precheck, processing
- Stage E: gap refill에서만 고객 query 성격이 뒤늦게 반영

이 구조에서는 상류에서 이미 generic pool이 만들어지기 때문에, downstream selection이 좋아도 최종 결과가 빈약하거나 편향되기 쉬웠습니다.

### 2. 공통 hot topic을 다룰 구조가 없었음

예를 들어 어떤 날 SK hynix 관점에서 중요한 공통 이슈가 있어도, 이를 개인화 결과와 충돌 없이 넣는 구조가 없었습니다.

기존에는 둘 중 하나가 되기 쉬웠습니다.

- 아예 빠짐
- 정적 query/keyword를 억지로 늘려서 과도하게 들어감

둘 다 품질이 좋지 않았습니다.

### 3. 중복/유사 이슈가 최종본에 남을 수 있었음

기존에도 title/semantic dedupe는 있었지만, 최종 selection 직전의 edition-level overlap-aware rerank는 없었습니다.

그래서 아래와 같은 문제가 남았습니다.

- 같은 이슈의 다른 각도 기사 2~3개가 함께 남는 문제
- shared topic을 넣으면 개인화 기사와 충돌하는 문제
- source/domain/country rebalance 이후 중복 억제 의도가 약해지는 문제

---

## 이번 변경의 목표

이번 단계의 목표는 단순히 기사 수를 늘리는 것이 아니라, 최종 뉴스레터 품질을 개선하는 것입니다.

핵심 목표는 아래와 같습니다.

- 고객 의도가 shortlist 이전 상류 풀에 실제로 반영되게 하기
- 공통 hot topic을 최대 2개까지 bounded insertion으로 넣되, low-affinity/saturated issue는 건너뛰기
- 최종 edition에서 같은 이슈/같은 각도 중복을 줄이기
- 이 모든 의사결정을 기존 observability/API에서 추적 가능하게 하기

---

## 실제 변경 사항

## 1. query-aware upstream pool 추가

`openclaw/agents/orchestrator.py`와 `openclaw/agents/news_collector.py`를 확장해서 Stage A에서 query-aware pool을 함께 만들도록 바꿨습니다.

바뀐 흐름은 아래와 같습니다.

- static source pool 수집
- 고객별 `searchQueries`, `needClusters`, `watch_companies` 기반 query plan 생성
- weighted custom query collection 실행
- static + query-aware 후보를 합친 뒤 dedupe
- 그 결과를 Stage A master pool로 사용

즉 이제 고객 intent가 shortlist/precheck 이전 upstream candidate formation에 반영됩니다.

## 2. phase2 helper 추가

`openclaw/agents/personalization_phase2.py`를 추가했습니다.

이 helper가 담당하는 일은 다음과 같습니다.

- 고객 query plan weight 계산
- SK hynix affinity 계산
- issue angle identity 생성
- shared hot-topic candidate 선택
- overlap-aware rerank 수행

이 로직을 따로 분리한 이유는 `orchestrator.py` 내부에서 hot-topic/rerank 계산까지 전부 섞이지 않게 하기 위해서입니다.

## 3. shared SK hynix hot-topic injection 추가

최종 selection 흐름에서 shared hot-topic 후보를 동적으로 선택하는 로직을 넣었습니다.

중요한 점은 아래와 같습니다.

- 하드코딩된 이벤트 리스트를 쓰지 않음
- corpus 안의 기사들에서 issue angle을 만들어 판단
- SK hynix relevance가 충분하지 않으면 skip
- 이미 같은 이슈가 edition에 있으면 skip
- 최대 2개까지만 허용

즉 “무조건 넣는” 기능이 아니라, 조건이 맞을 때만 bounded insertion 되는 구조입니다.

## 4. overlap-aware final rerank 추가

최종 selection 직전에 overlap-aware rerank를 추가했습니다.

삽입 위치:

- `_select_for_customer()`
- refill loop
- shared hot-topic injection
- overlap-aware rerank
- `_rebalance_domain_bias()`
- `_apply_country_floor()`
- `_fill_minimum_articles()`

이 단계에서 같은 `issue_angle_id`를 가진 후보가 final edition을 과하게 차지하지 않도록 줄입니다.

즉 기존 semantic/title dedupe보다 더 마지막 edition 단계의 중복 억제를 추가한 셈입니다.

## 5. observability 확장

이번 phase2 의사결정도 기존 ledger/API를 통해 볼 수 있게 확장했습니다.

추가적으로 보이는 정보 예시:

- upstream query-aware contribution 개수
- shared hot-topic inject/skip 이유
- rerank 결과와 personalization share
- selected item별 `originType`, `originDetail`, `issueAngleId`

즉 API에서 “왜 이 기사가 들어왔는지”를 이전보다 더 설명할 수 있습니다.

## 6. 정책 추가

`newsletter_agent.json`에 `phase2` 설정을 추가했습니다.

현재 기본 정책:

- `queryAwareUpstreamPool.enabled: true`
- `sharedSkhynixHotTopic.enabled: true`
- `sharedSkhynixHotTopic.capPerEdition: 2`
- `overlapAwareFinalRerank.enabled: true`
- `overlapAwareFinalRerank.personalizationShareFloor: 0.75`

즉 이번 기능은 정책 기반으로 켜고 끌 수 있게 만들었습니다.

---

## 운영 관점에서 좋아진 점

### 1. 고객 의도가 더 일찍 반영됨

이전에는 Stage E refill에서만 드러났던 고객 query 의도가 이제 Stage A 상류 pool에 직접 반영됩니다.

### 2. 공통 중요 이슈를 안전하게 다룰 수 있음

SK hynix 입장에서 중요한 공통 hot topic을 개인화 결과와 완전히 별개로 떼어놓지 않고, bounded shared editorial layer로 다룰 수 있게 됐습니다.

### 3. 최종 뉴스레터 중복이 줄어듦

같은 이슈의 여러 기사나 거의 같은 angle의 기사가 final edition에 동시에 남는 가능성이 줄어들었습니다.

### 4. 의사결정이 설명 가능해짐

이번 기능도 기존 observability 위에 올라가므로, 아래를 계속 확인할 수 있습니다.

- 이 기사는 static source였는지 query-aware source였는지
- shared hot-topic으로 들어왔는지
- 왜 skip됐는지
- rerank 이후 personalization share가 얼마였는지

---

## 검증 내용

이번 변경은 아래 순서로 검증했습니다.

### 1. phase2 테스트

```bash
python3 -m unittest discover -s tests/openclaw -p "test_personalization_phase2*.py"
```

검증 범위:

- query plan weighting
- Stage A query-aware upstream pool 형성
- high-affinity user hot-topic injection
- low-affinity / saturated issue skip
- overlap-aware rerank
- personalization share 유지

### 2. observability 회귀 테스트

```bash
python3 -m unittest discover -s tests/openclaw -p "test_observability_*.py"
```

검증 범위:

- phase2 customer decision 필드 노출
- selected item의 `originType`, `originDetail`, `issueAngleId`
- 기존 API payload의 additive 확장 유지

### 3. 문법/설정 검증

```bash
python3 -m py_compile openclaw/agents/personalization_phase2.py openclaw/agents/news_collector.py openclaw/agents/observability.py openclaw/api_server.py
python3 -m json.tool newsletter_agent.json
```

### 4. 수동 QA

실제로 합성 시나리오를 실행해서 아래를 확인했습니다.

- Stage A가 `staticCount`, `queryAwareCount`, `queryPlanCount`를 가진 upstream metadata를 만듦
- shared hot topic이 `eligible_hot_topic`으로 inject 됨
- final selected item에 `static_source`, `query_aware_source`, `shared_topic_injected`가 함께 남음
- API `phase2` payload에 upstream/hot topic/rerank 결과가 노출됨

---

## 이번 변경이 해결한 범위

이번 단계에서 직접 해결한 것은 아래입니다.

- 고객 query 기반 상류 pool 확장
- bounded shared SK hynix hot-topic injection
- final overlap-aware rerank
- phase2 observability 확장

반대로 아직 이번 단계의 직접 범위는 아닙니다.

- cross-customer overlap enforcement
- 완전한 embedding/ML 기반 recommender 시스템
- 월간 research lane의 본격적 재설계
- web UI/dashboard 추가

즉 이번 작업은 기존 `openclaw` 구조를 유지하면서 실제 품질 개선을 넣은 2단계입니다.

---

## 참고 파일

- `openclaw/agents/personalization_phase2.py`
- `openclaw/agents/news_collector.py`
- `openclaw/agents/orchestrator.py`
- `openclaw/agents/observability.py`
- `tests/openclaw/test_personalization_phase2.py`
- `tests/openclaw/test_observability_api.py`
- `newsletter_agent.json`

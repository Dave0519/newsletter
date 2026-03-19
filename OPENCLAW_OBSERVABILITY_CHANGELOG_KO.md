# Openclaw 관측성 개선 문서

## 문서 목적

이 문서는 이번 `openclaw` 운영 관측성 개선 작업에서 무엇이 문제였고, 무엇을 어떻게 바꿨는지, 그리고 실제로 어디까지 검증했는지를 정리합니다.

`README.md`가 사용법 중심이라면, 이 문서는 변경 배경과 운영 관점의 의미를 설명하는 문서입니다.

---

## 기존 문제점

### 1. 파이프라인이 돌아가도 한눈에 상태를 보기 어려웠음

기존 `openclaw`는 Stage A-F 흐름과 checkpoint 파일 자체는 있었지만, 운영자가 바로 읽을 수 있는 표준 상태 모델이 없었습니다.

즉, 아래 질문에 답하기가 어려웠습니다.

- 지금 어떤 run이 돌고 있는가
- 각 고객별로 어느 stage까지 갔는가
- 어느 stage에서 몇 건이 줄었는가
- 최종 발행 실패가 gate 문제인지 delivery 문제인지

기존 정보는 `newsletter_agent_state.json`, checkpoint JSON, 말단 result object, cron 상태 등에 흩어져 있었고, 한번에 모아보는 얇은 API가 없었습니다.

### 2. 정보 손실 지점을 추적하기 어려웠음

운영에서 가장 중요한 질문 중 하나는 “어디에서 정보가 빠졌는가”인데, 기존 구조는 이 부분이 약했습니다.

예를 들어 아래는 일부 알 수 있었지만, 연결된 형태로 남지 않았습니다.

- Stage A 이후 dedupe로 줄었는지
- shortlist 단계에서 고객 relevance 때문에 빠졌는지
- precheck에서 탈락했는지
- 본문 추출/LLM 처리에서 사라졌는지
- 최종 selection에서 밀렸는지
- 발행 gate에서 막혔는지

결과적으로 “최종 뉴스레터가 왜 빈약해졌는지”를 구조적으로 설명하기 어려웠습니다.

### 3. 소스별 상태를 비교해서 보기 어려웠음

수집 단계에서는 소스별 품질 편차가 중요한데, 기존 구조에서는 소스/domain 기준으로 아래를 한 번에 보기가 어려웠습니다.

- 얼마나 수집됐는지
- shortlist까지 얼마나 살아남았는지
- precheck/processing/select 단계에서 얼마나 줄었는지
- 특정 소스가 지속적으로 손실을 많이 유발하는지

즉 “어느 소스가 실제로 유효한 공급원인지”를 운영자가 빠르게 판단하기 어려웠습니다.

### 4. 개인화 결과를 고객별로 설명하기 어려웠음

최종 고객별 선택 결과는 있었지만, 왜 그 기사가 선택됐고 무엇이 빠졌는지에 대한 운영용 의사결정 로그가 부족했습니다.

그래서 아래와 같은 설명이 어려웠습니다.

- 이 고객은 왜 이 기사 세트를 받았는가
- need gap은 있었는가
- gate는 통과했는가
- delivery는 성공했는가

### 5. 운영 확인용 인터페이스가 부족했음

웹 UI를 만들 필요는 없었지만, 최소한 API 기반으로 현재 상태를 확인하는 표면은 필요했습니다. 기존에는 파일을 직접 열어보거나 실행 결과를 추적해야 해서 운영 효율이 떨어졌습니다.

---

## 이번 변경의 목표

이번 작업은 기존 파이프라인을 갈아엎는 것이 아니라, 현재 `openclaw` 구조를 유지한 채 운영 관측성을 추가하는 데 목적이 있었습니다.

핵심 목표는 아래와 같습니다.

- 실행 중/완료된 run을 API로 확인할 수 있게 만들기
- stage별 in/out/loss를 기록하기
- 기사/후보가 어디서 탈락했는지 reason code와 함께 남기기
- 고객별 최종 선택과 gate/delivery 결과를 기록하기
- source/domain 기준 손실 현황을 요약하기
- 웹페이지 없이도 운영자가 drill-down 할 수 있게 하기

---

## 실제 변경 사항

## 1. 운영 ledger 추가

기존 runtime contract를 유지하면서, 아래 ledger를 추가로 기록하도록 했습니다.

- `newsletter_stage_runs.jsonl`
- `newsletter_loss_events.jsonl`
- `newsletter_customer_decisions.jsonl`
- `newsletter_source_health.jsonl`

기존 파일도 계속 사용합니다.

- `newsletter_agent_state.json`
- `newsletter_runs.jsonl`
- `newsletter_audit.jsonl`

즉, 기존 구조를 깨지 않고 운영 관측용 기록만 additive 하게 붙였습니다.

## 2. 관측 전용 helper 추가

`openclaw/agents/observability.py`를 추가했습니다.

이 모듈이 담당하는 일은 아래와 같습니다.

- run/stage/loss/customer/source ledger 읽기/쓰기
- 기사 후보의 stable id 계산
- stage 전후 후보 차이 계산
- source/domain 기준 손실 스냅샷 생성
- 특정 run 기준 summary 조립

이 모듈을 따로 둔 이유는 `orchestrator.py`에 ledger 조합 로직까지 모두 섞지 않기 위해서입니다.

## 3. orchestrator에 stage/loss 기록 추가

`openclaw/agents/orchestrator.py`에 additive hook을 넣었습니다.

주요 반영 지점:

- Stage A: shared pool 생성 후 dedupe 손실 기록
- Stage B: shortlist 전후 차이 기록
- Stage C: precheck 전후 차이 기록
- Stage D: processing 전후 차이 기록
- Stage E: final selection 전후 차이 기록
- Stage F: gate 통과 여부와 publication 결과 기록

또한 아래도 함께 기록합니다.

- 고객별 최종 selection decision
- source/domain 기준 source health snapshot
- 기존 run/audit log에 `runId`, `totalResearch` 등 추가 필드 포함

중요한 점은 publish path를 바꾸지 않고, 실패 시에도 ledger write가 본 흐름을 깨지 않도록 best-effort 성격으로 넣었다는 점입니다.

## 4. 읽기 전용 API 추가

`openclaw/api_server.py`를 추가했습니다.

새 프레임워크를 들이지 않고 Python 표준 라이브러리 `http.server` 기반으로 얇게 구현했습니다.

지원하는 주요 endpoint:

- `/health`
- `/api/summary`
- `/api/runs`
- `/api/stages`
- `/api/losses`
- `/api/customers`
- `/api/sources`
- `/api/cron`
- `/api/checkpoints`

이 API는 실행을 바꾸지 않고, 이미 기록된 ledger와 checkpoint를 읽어서 운영자가 상태를 조회하는 용도입니다.

## 5. README 개선

루트 `README.md`를 한국어로 정리했고, 아래 내용을 반영했습니다.

- API 실행 방법
- endpoint 목록
- `curl` 예시
- 실제 응답 예시
- 생성되는 ledger 설명
- 테스트/검증 방법

즉, 문서만 읽어도 “어떻게 띄우고 무엇이 보이는지”를 이해할 수 있도록 바꿨습니다.

---

## 운영 관점에서 좋아진 점

이번 변경 이후 운영자는 최소한 아래를 API와 ledger 기준으로 확인할 수 있습니다.

### 1. run 중심 상태 확인

`/api/summary`로 현재 run 기준 전체 상태를 볼 수 있습니다.

- 현재 state
- run log
- stage별 요약
- loss 이벤트
- customer decision
- source health
- checkpoint 경로
- cron 상태

### 2. 손실 지점 확인

`/api/losses`를 통해 어떤 후보가 어느 stage에서 왜 떨어졌는지 볼 수 있습니다.

예:

- `precheck_rejected`
- `processing_dropped`
- `final_selection_drop`
- `publish_gate_blocked`

### 3. 고객별 최종 결과 확인

`/api/customers`에서 고객별 coverage, gate 결과, delivery 결과, 최종 selected 후보를 볼 수 있습니다.

### 4. 소스별 손실 확인

`/api/sources`에서 source/domain 기준으로 collected -> shortlisted -> prechecked -> cleaned -> selected 흐름을 요약해서 볼 수 있습니다.

### 5. checkpoint drill-down

`/api/checkpoints?...&include_data=1`로 checkpoint 파일 경로만 보는 것이 아니라 실제 data까지 함께 볼 수 있습니다.

---

## 이번 변경이 해결한 범위

이번 작업으로 해결한 것은 “운영 가시성”과 “설명 가능성”입니다.

즉, 아래는 이번에 해결했습니다.

- 실행 상태를 API로 보는 문제
- stage별 손실 추적 문제
- source/customer 기준 drill-down 문제
- README 부재 문제

반대로 아래는 아직 이번 작업의 직접 범위는 아닙니다.

- 개인화 상류 풀을 고객 query 기반으로 재설계하는 것
- SK hynix 공통 hot topic을 동적으로 주입하는 것
- overlap-aware reranking을 실제 selection 로직에 넣는 것
- monthly research lane을 본격적으로 실행 경로화하는 것

이 항목들은 승인된 계획에는 포함되어 있지만, 이번 구현은 그 전에 필요한 관측 기반을 먼저 추가한 단계입니다.

---

## 검증 내용

이번 변경은 코드 작성 후 실제로 아래를 검증했습니다.

### 1. 단위 테스트

```bash
python3 -m unittest discover -s tests/openclaw -p "test_observability_*.py"
```

검증 대상:

- ledger append/read 동작
- source health snapshot 계산
- API payload 조립
- checkpoint data 포함 응답

### 2. 수동 API 검증

실제로 서버를 띄우고 아래 endpoint를 호출해 응답 JSON을 확인했습니다.

- `/health`
- `/api/summary?run_id=qa-run`
- `/api/losses?customer_id=cust-a`
- `/api/checkpoints?run_id=qa-run&customer_id=cust-a&include_data=1`

### 3. publish switch 차단 시 ledger 기록 확인

publish switch를 끈 복사본 환경에서 `orchestrator.run()`을 실제로 실행해 아래 파일이 기록되는 것을 확인했습니다.

- `newsletter_runs.jsonl`
- `newsletter_audit.jsonl`

즉, 발행이 차단돼도 운영 trace는 남는다는 점을 확인했습니다.

---

## 커밋 정보

이번 작업은 아래 두 커밋으로 정리되었습니다.

- `7742796` `feat(openclaw): add observability ledgers and read API`
- `f786fb4` `docs(readme): document observability API in Korean`

작업 브랜치:

- `feat/openclaw-observability-api`

---

## 참고 파일

구현 핵심 파일:

- `openclaw/agents/orchestrator.py`
- `openclaw/agents/observability.py`
- `openclaw/api_server.py`
- `tests/openclaw/test_observability_store.py`
- `tests/openclaw/test_observability_api.py`
- `README.md`

계획 문서:

- `.sisyphus/plans/openclaw-observability-redesign.md`

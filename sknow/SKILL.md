---
name: sknow
description: Run clue_news_collector wide-collect and bridge to clue_letter_dev2 using shared total-pool, with optional parallel rendering and optional send step.
---

# CLUE News Collector + Dev2 Shared-Pool Pipeline

이 스킬은 `clue_news_collector`의 수집 결과를 직접 유저별로 고정 라우팅하지 않고,
`shared total_news` 풀로 먼저 적재한 뒤 `clue_letter_dev2`에서 유저별 니즈 매칭을 거쳐 렌더링합니다.

## 기본 동작

1. `clue_news_collector` 실행 (`run_wide_collect.py`)
2. 수집 요약(`wide_collect.summary.json`)의 `run_id`를 기반으로 raw 후보를 통합
3. 후보를 중복 제거해 `clue_letter_dev2/data/shared/daily news/total_news_<YYYY-MM-DD>.jsonl`로 작성
4. `clue_letter_dev2`의 `service.py run-all --no-send`로 유저별 `daily news` 생성 및 HTML 작성
5. 필요 시 `service.py run-all`으로 발송 단계를 분리 실행(기본은 보내기)

## 핵심 정책

- **원본 후보 저장소도 덮어쓰기(재수집 오버라이트) 기반**
  - 기본적으로 `sknow`는 실행 전 `data/ingest/raw`, `data/ingest/curated`, `data/known_urls`(ledger) 를 비워서,
    이전 누적 이력을 다음 수집에 섞이지 않게 처리합니다.
  - `--no-overwrite-candidates` 옵션으로 비우기 동작을 끌 수 있습니다.
- **shared-pool 중심(비하드코딩) 처리**
  - collector는 유저별 고정 출력 대신 공통 풀에 적재
  - dev2가 `CLUE_SHARED_POOL`/공유 풀 정책으로 유저 니즈(`needs`) 기반 매칭
- **병렬 렌더링 + 순차 발송**
  - `sknow` 실행 시 렌더링은 병렬(`--parallel-workers`) 가능
  - 발송은 기본적으로 독립 단계로 분리되어 순차 실행
- **호환성 보강**
  - `collector`/`service` 하위 실행은 Python 실행기(`--python` 또는 `SKNOW_PYTHON`)를 통해 통일
  - 권장 Python: 3.12 이상
  - collector 경로/스크립트는 인자/환경변수로 외부 주입

## 동작 규칙

- 유저를 특정 사용자 하나로 하드코딩해서 고정하지 않습니다.
- `--user-code`는 레거시 단일 사용자 실행용으로만 사용합니다.
- 기본 동작은 전체 active user 대상(`run-all`)입니다.

## 환경 설정

- `OPENAI_API_KEY`가 없으면 ` /Users/davechoi/.openclaw/.env `,
  ` /Users/davechoi/.openclaw/workspace/openclaw/.env `에서 보완 로드합니다.
- `OPENAI_MODEL` 기본값: `gpt-5-mini`

## 사용 예시

```bash
cd /path/to/skills/sknow
python3 scripts/run_news_pipeline.py \
  --collect \
  --top-items 20 \
  --parallel-workers 3 \
  --python /usr/local/bin/python3.12 \
  --send
```

- `--skip-collect`: 최신 shared pool 재사용해 유저별 렌더링만 수행
- `--collect-only`: shared pool까지 만들고 중단
- `--collect --overwrite-candidates`: 수집 전 원본 후보 저장소 정리 후 수집(기본)
- `--no-overwrite-candidates`: 원본 후보 정리 안하고 기존 누적 유지(권장 없음)
- `--no-send`: 생성만 하고 발송 생략
- `--parallel-workers`: 유저별 렌더링 병렬도(예: 2~4)
- `--user-code`: 레거시 단일 사용자 모드(run)
- `--python`: collector/service 실행용 Python 인터프리터 지정(기본: 시스템에서 3.12+ 탐색)

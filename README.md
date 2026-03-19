# newsletter

이 저장소는 두 가지 실행 경로를 가진 Python 뉴스레터 워크스페이스입니다.

- `clue_letter/`: 사용자 등록, 니즈 관리, HTML 작성까지 포함한 서비스형 실행 경로
- `openclaw/`: Stage A-F 배치 파이프라인과 운영 관측용 ledger/API 경로

## Openclaw 운영 관측 API

웹페이지 없이도 현재 처리 상태를 확인할 수 있도록 `openclaw`용 읽기 전용 API를 추가했습니다.

서버 실행:

```bash
python3 openclaw/api_server.py --host 127.0.0.1 --port 8765
```

주요 엔드포인트:

- `/health`
- `/api/summary`
- `/api/runs`
- `/api/stages`
- `/api/losses`
- `/api/customers`
- `/api/sources`
- `/api/cron`
- `/api/checkpoints?run_id=<RUN_ID>&customer_id=<CUSTOMER_ID>&include_data=1`

예시 호출:

```bash
curl "http://127.0.0.1:8765/api/summary"
curl "http://127.0.0.1:8765/api/stages?run_id=<RUN_ID>"
curl "http://127.0.0.1:8765/api/losses?run_id=<RUN_ID>&customer_id=<CUSTOMER_ID>"
curl "http://127.0.0.1:8765/api/checkpoints?run_id=<RUN_ID>&customer_id=<CUSTOMER_ID>&include_data=1"
```

## API 응답 예시

`/health`

```json
{
  "ok": true
}
```

`/api/summary?run_id=qa-run`

```json
{
  "runId": "qa-run",
  "state": {
    "run_id": "qa-run",
    "stage": "F"
  },
  "runs": [
    {
      "runId": "qa-run",
      "customerId": "cust-a",
      "status": "ok",
      "totalScan": 3
    }
  ],
  "stages": [
    {
      "runId": "qa-run",
      "stageId": "B",
      "customerId": "cust-a",
      "inCount": 10,
      "outCount": 3,
      "lossCount": 7,
      "status": "completed"
    }
  ],
  "losses": [
    {
      "runId": "qa-run",
      "stageId": "C",
      "customerId": "cust-a",
      "reasonCode": "precheck_rejected",
      "title": "drop me",
      "domain": "example.com"
    }
  ],
  "customerDecisions": [
    {
      "runId": "qa-run",
      "customerId": "cust-a",
      "gateOk": true,
      "coverage": {
        "need_gap": false
      }
    }
  ],
  "sourceHealth": [
    {
      "runId": "qa-run",
      "customerId": "cust-a",
      "sources": [
        {
          "domain": "example.com",
          "counts": {
            "collected": 3,
            "selected": 1
          }
        }
      ]
    }
  ]
}
```

`/api/losses?customer_id=cust-a`

```json
{
  "items": [
    {
      "runId": "qa-run",
      "stageId": "C",
      "customerId": "cust-a",
      "reasonCode": "precheck_rejected",
      "title": "drop me",
      "url": "https://example.com/drop",
      "domain": "example.com"
    }
  ]
}
```

`/api/checkpoints?run_id=qa-run&customer_id=cust-a&include_data=1`

```json
{
  "runId": "qa-run",
  "paths": {
    "shortlists": {
      "cust-a": {
        "path": "/tmp/.../shortlist.json",
        "data": [
          {
            "title": "HBM news",
            "url": "https://example.com/hbm"
          }
        ]
      }
    }
  }
}
```

## 생성되는 운영 ledger

기존 runtime contract는 유지하고, 아래 ledger를 루트에 추가로 기록합니다.

- `newsletter_agent_state.json`
- `newsletter_runs.jsonl`
- `newsletter_audit.jsonl`
- `newsletter_stage_runs.jsonl`
- `newsletter_loss_events.jsonl`
- `newsletter_customer_decisions.jsonl`
- `newsletter_source_health.jsonl`

이 파일들로 다음을 바로 확인할 수 있습니다.

- 어떤 run이 실행됐는지
- 각 stage에서 몇 건이 들어오고 몇 건이 남았는지
- 어떤 기사/후보가 어디에서 탈락했는지
- 고객별 최종 선택과 gate 결과가 어땠는지
- source/domain 기준으로 어느 단계에서 손실이 컸는지

## 검증

관측 레이어 전용 테스트:

```bash
python3 -m unittest discover -s tests/openclaw -p "test_observability_*.py"
```

수동 smoke test:

```bash
python3 openclaw/api_server.py --host 127.0.0.1 --port 8765
curl "http://127.0.0.1:8765/api/summary"
```

publish switch가 꺼진 상태에서도 `newsletter_runs.jsonl` / `newsletter_audit.jsonl`이 실제로 기록되는지 별도 확인했습니다.

## 기존 서비스 문서

서비스형 실행 흐름은 `clue_letter/README.md`를 참고하면 됩니다.

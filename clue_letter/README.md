# clue_letter (service)

브라우저 릴레이 기반 뉴스레터를 위한 실제 운영 서비스 구조.
테스트용 샘플이 아니라 **실구동용 모듈**로 정리했습니다.

## 목표 (요청사항 반영)
1. 사용자 별 니즈 관리(Needs Agent)
2. 브라우저 기반 기사 수집(24시간 내/키워드 커버리지, 사용자별 다중 니즈 탐색)
3. 수집 데이터(제목, 원문, 요약, URL, country) 저장
4. 뉴스레터 작성(공식 템플릿, No 카운팅, 해시태그, 날짜, 사용자 코드, 요약)
5. 슈퍼 에이전트가 Needs/Collection/Writing/Delivery 조율

## 핵심 특징
- `collection`은 브라우저 릴레이 어댑터(`agents/browser_adapter.py`)를 사용.
- 기사 원문은 브라우저에서 추출한 텍스트를 바탕으로 처리.
- 제목/요약은 한국어 정제(LLM 우선, API 미설정 시 폴백).
- 사용자 폴더 규격:
  - `data/{사용자명}/daily news/{YYYY-MM-DD}.jsonl`
  - `data/{사용자명}/history/titles.txt`

## 실행 방식

### 사용자 등록
```bash
python3 clue_letter/service.py register --name "lcs" --interests "AI 인프라" "반도체" "데이터센터"
```
(메일은 기본 `bonggyu1.choi@sk.com`)

### 니즈 변경
```bash
python3 clue_letter/service.py interests --user-code <USER_CODE> --interests "AI 인프라" "반도체" "클라우드"
```

### 단일 사용자 실행
```bash
python3 clue_letter/service.py run --user-code <USER_CODE> --dry
python3 clue_letter/service.py run --user-code <USER_CODE>
```

### 전체 사용자 실행
```bash
python3 clue_letter/service.py run-all --dry
python3 clue_letter/service.py run-all
```

## 비고
- 브라우저 릴레이는 `openclaw browser` CLI로 동작.
- 동작 전 `openclaw browser start` 또는 스크립트 최초 호출 시 자동 `start`.
- 수집 최소 건수 기본 8개.
- 템플릿은 `templates/CLUE_TEMPLATE_OFFICIAL.html` 사용.

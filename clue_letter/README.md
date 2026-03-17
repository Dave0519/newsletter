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
- 기사 원문은 브라우저에서 추출한 텍스트를 뽑기 전에, 제목/스니펫 메타데이터 기반으로 1차 사전 선별 후 본문 fetch+요약 처리(속도 최적화).
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

## 빠른 자동 커밋/푸시

정책/실행 구조 코드만 변경될 때는 아래로 즉시 반영 가능:

```bash
cd /Users/davechoi/.openclaw/workspace/clue_letter
./auto_sync_clue.sh "chore(clue_letter): ..."
```

- 변경 감지 대상: `clue_letter/` 하위 정책·실행 파일(agents/service/템플릿/core_rss/런처/테스트 스크립트)
- 제외 대상: `clue_letter/data/`, `clue_letter/logs/` (런타임 산출물)
- 기본 동작: 변경 있으면 `git add` → `git commit` → `git push origin main`
- 변경 없으면 `No clue_letter policy/runtime changes to sync.`로 종료

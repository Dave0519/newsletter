# CORE RSS 상태 저장 (2026-03-17 기준)

## ✅ 현재 추출 OK (즉시 사용 가능)
- Ars Technica AI: https://arstechnica.com/tag/ai/feed/
- VentureBeat AI: https://www.venturebeat.com/category/ai/feed/
- Wccftech: https://www.wccftech.com/feed/
- Azure Blog: https://azure.microsoft.com/en-us/blog/feed/
- WSJ World: https://feeds.a.dj.com/rss/RSSWorldNews.xml

## ⚠️ 추출 불안정/실패 (구조 확인 필요)
- TechCrunch AI: https://techcrunch.com/tag/ai/feed/
- The Verge AI: https://www.theverge.com/ai/rss/index.xml
- MIT Tech Review AI: https://www.technologyreview.com/feed/
- Electronic Design: https://www.electronicdesign.com/feed/
- AnandTech news: https://www.anandtech.com/rss/news/
- Google Cloud Blog: https://cloud.google.com/blog/rss
- AWS What's New: https://aws.amazon.com/about-aws/whats-new/recent/feed/
- CNBC Tech: https://www.cnbc.com/id/10000113/device/rss/rss.html
- Reuters Tech: https://www.reuters.com/arc/outboundfeeds/sitemap/?output=atom
- FT: https://www.ft.com/?format=rss

## 살릴 수 있는 후보 (우선순위)
1. The Verge: 경로 자체가 RSS가 아닌 페이지로 응답되는 것으로 보임. 대체 FEED URL 재확인 필요.
2. TechCrunch: 사이트 정책/리디렉션 의심. `https://techcrunch.com/category/artificial-intelligence/feed/` 형태로 대체 후보 점검 필요.
3. Google Cloud / AWS / CNBC / FT / Reuters: 직접 브라우저 페이지 렌더링에서 XML 항목 추출 실패가 섞여 있어, RSS가 아닌 구조 또는 클라우드플레어/차단/헤더 의존일 가능성 큼.
4. AnandTech / Electronic Design: 피드가 존재해도 렌더링/포맷이 브라우저 추출기에 맞지 않을 수 있어, raw URL fetch(XML 파서) 경로 추가 시 회복 가능성 있음.

## 2026-03-18 신규 확장 제안 반영
- 추가 활성화(확장):
  - TechCrunch AI (category): https://techcrunch.com/category/artificial-intelligence/feed/
  - The Verge: https://www.theverge.com/rss/index.xml
  - MIT Tech Review: https://www.technologyreview.com/feed/
  - Wired: https://www.wired.com/feed/category/gear/latest/rss
  - ZDNet: https://www.zdnet.com/topic/technology/news/rss.xml
  - CNET Tech: https://www.cnet.com/rss/news/
  - 연합뉴스: https://www.yna.co.kr/rss/all.xml
  - 전자신문: https://www.etnews.com/rss/all.xml

- 이전 실패 후보 중 대체 URL/정상 후보로 이동 시도:
  - TechCrunch tag feed(기존) -> category feed로 교체
  - The Verge ai feed(기존 실패) -> 공용 RSS 경로로 교체
  - MIT TechReview 기존 실패 경로 -> /feed/ 경로로 교체

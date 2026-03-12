from __future__ import annotations
from pathlib import Path
import hashlib
import re

COUNTRY_LABEL={"KR":"한국","US":"미국","CN":"중국","TW":"대만","GLOBAL":"글로벌"}

class NewsletterBuilder:
    def __init__(self, template_path: str, country_order: list[str]):
        self.template_path=template_path
        self.country_order=country_order

    def template_fingerprint(self):
        p=Path(self.template_path)
        raw=p.read_bytes()
        return str(p.resolve()), hashlib.sha256(raw).hexdigest()

    def validate(self, scan: dict[str,list[dict]], research: list[dict], min_scan: int=10, min_research: int=0):
        total=sum(len(v) for v in scan.values())
        errs=[]
        if total < min_scan:
            errs.append(f"global_scan<{min_scan}")
        if len(research) < min_research:
            errs.append(f"research<{min_research}")
        return len(errs)==0, errs

    def build(self, scan: dict[str,list[dict]], research: list[dict], issue_date: str, issue_number: str="001", serial_number: str="", needs_hashtags: str="", brand_mark: str="SK hynix", global_scan_intro: str=""):
        html=Path(self.template_path).read_text(encoding='utf-8')
        country_block=self._extract_block(html,'{{#COUNTRIES}}','{{/COUNTRIES}}')
        rendered=[]
        for c in self.country_order:
            items=scan.get(c,[])
            if not items: continue
            row=country_block.replace('{{COUNTRY_NAME}}',COUNTRY_LABEL.get(c,c))
            article_block=self._extract_block(row,'{{#ARTICLES}}','{{/ARTICLES}}')
            rows=[]
            for a in items:
                r=article_block
                r=r.replace('{{ARTICLE_TITLE}}',self._esc(a.get('title_ko') or a.get('title','')))
                r=r.replace('{{ARTICLE_SUMMARY}}',self._esc(a.get('description','')))
                r=r.replace('{{ARTICLE_PRACTICAL_IMPLICATION}}',self._esc(a.get('practical_implication','')))
                r=r.replace('{{ARTICLE_LINK}}',a.get('url','#'))
                rows.append(r)
            row=self._replace_block(row,'{{#ARTICLES}}','{{/ARTICLES}}','\n'.join(rows))
            rendered.append(row)
        html=self._replace_block(html,'{{#COUNTRIES}}','{{/COUNTRIES}}','\n'.join(rendered))
        bullets=[]
        for c in self.country_order:
            for a in scan.get(c,[]):
                d=' '.join((a.get('description') or '').split())
                if d: bullets.append('• '+self._esc(d[:220]))
                if len(bullets)>=5: break
            if len(bullets)>=5: break
        summary='<br><br>'.join(bullets) if bullets else '오늘은 본문 추출 가능한 주요 기사가 부족했습니다.'
        for k,v in {
            '{{ISSUE_DATE}}':issue_date,'{{ISSUE_NUMBER}}':issue_number,'{{SERIAL_NUMBER}}':serial_number,
            '{{NEEDS_HASHTAGS}}':needs_hashtags,'{{BRAND_MARK}}':brand_mark,'{{GLOBAL_SCAN_INTRO}}':global_scan_intro,
            '{{CORE_DESCRIPTION}}':summary
        }.items(): html=html.replace(k,v)
        html=re.sub(r"CLUE 데일리 브리핑\s*·\s*[^·<]+\s*·", f"CLUE 데일리 브리핑 · {issue_date} ·", html)
        return html

    @staticmethod
    def _esc(s:str)->str:
        return (s or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    @staticmethod
    def _extract_block(text,start,end):
        i=text.find(start); j=text.find(end)
        if i==-1 or j==-1 or j<i: return ''
        return text[i+len(start):j]
    @staticmethod
    def _replace_block(text,start,end,replacement):
        i=text.find(start); j=text.find(end)
        if i==-1 or j==-1 or j<i: return text
        return text[:i]+replacement+text[j+len(end):]

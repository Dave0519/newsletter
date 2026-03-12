from __future__ import annotations
import subprocess

class DeliveryManager:
    def __init__(self, _email_cfg: dict|None=None, _telegram_cfg: dict|None=None):
        self.email_cfg=_email_cfg or {}

    def deliver(self, html: str, issue_date: str, recipient: str):
        if not recipient:
            return {"email":False,"telegram":False}
        subject='[CLUE] AI BRIEFING | 오늘의 글로벌 핵심 동향'
        cmd=['gog','gmail','send','--to',recipient,'--subject',subject,'--body-html',html or '<p>No content</p>']
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
            return {"email": r.returncode==0, "telegram":False, "stdout":r.stdout[-500:], "stderr":r.stderr[-500:]}
        except Exception:
            return {"email":False,"telegram":False}

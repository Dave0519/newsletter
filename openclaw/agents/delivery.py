from __future__ import annotations

from pathlib import Path
import subprocess


class DeliveryManager:
    def __init__(self, _email_cfg: dict | None = None, _telegram_cfg: dict | None = None):
        self.email_cfg = _email_cfg or {}

    def deliver(self, html: str, issue_date: str, recipient: str):
        if not recipient:
            return {"email": False, "telegram": False}

        tmp = Path("/tmp") / f"clue_{issue_date.replace('.', '')}.html"
        payload = html or "<html><body><p>No content</p></body></html>"
        tmp.write_text(payload, encoding="utf-8")
        subject = "[CLUE] AI BRIEFING | 오늘의 글로벌 핵심 동향"

        cmd = ["gog", "gmail", "send", "--to", recipient, "--subject", subject, "--body-html", payload]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            ok = (r.returncode == 0)
            return {"email": ok, "telegram": False, "stdout": r.stdout[-500:], "stderr": r.stderr[-500:]}
        except Exception:
            return {"email": False, "telegram": False}

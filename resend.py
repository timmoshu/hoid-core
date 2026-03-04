import logging
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
_API_KEY = os.getenv("RESEND_API_KEY")
_FROM_EMAIL = os.getenv("FROM_EMAIL")


async def send_email(to: str, subject: str, body: str, html: str = None, reply_to: str = None, extra_headers: dict = None):
    payload = {
        "from": _FROM_EMAIL,
        "to": [to],
        "subject": subject,
    }
    if body:
        payload["text"] = body
    if html:
        payload["html"] = html
    if reply_to is not None:
        payload["reply_to"] = reply_to
    if extra_headers:
        payload["headers"] = extra_headers
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if r.status_code >= 400:
        logger.error("Resend API error %s: %s", r.status_code, r.text)
        r.raise_for_status()

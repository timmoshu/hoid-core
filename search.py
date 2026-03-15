import os
from html.parser import HTMLParser

import httpx

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
FETCH_CHAR_LIMIT = 12000
JINA_BASE = "https://r.jina.ai/"

_JUNK_PATTERNS = ["Loading...", "not yet fully loaded", "Just a moment"]


def _is_usable_content(text: str) -> bool:
    """Return False if text is too short or contains known junk patterns."""
    if len(text) < 200:
        return False
    if len(text) < 500:
        return not any(p in text for p in _JUNK_PATTERNS)
    return True


class _TextExtractor(HTMLParser):
    """Minimal HTML → plain text extractor for direct-fetched pages."""
    _SKIP = {"script", "style", "noscript", "head"}

    def __init__(self):
        super().__init__()
        self._parts = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


async def web_search(query: str) -> str:
    if not TAVILY_API_KEY:
        return "Web search unavailable: TAVILY_API_KEY not configured."
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 8,
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        parts = []
        if data.get("answer"):
            parts.append(f"Summary: {data['answer']}\n")
        for r in data.get("results", []):
            parts.append(f"**{r['title']}** ({r['url']})\n{r.get('content', '')}")
        return "\n\n".join(parts) if parts else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


async def _fetch_via_jina(url: str) -> str:
    """Fetch via Jina.ai — handles Cloudflare, JS rendering, returns clean markdown."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(JINA_BASE + url)
    r.raise_for_status()
    return r.text


async def fetch_url(url: str) -> str:
    try:
        content = ""

        # Tier 1: direct httpx
        try:
            async with httpx.AsyncClient(
                timeout=20,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; hoid-core/1.0)"},
            ) as client:
                r = await client.get(url)
            if r.status_code == 403:
                raise httpx.HTTPStatusError("403", request=r.request, response=r)
            r.raise_for_status()
            parser = _TextExtractor()
            parser.feed(r.text)
            content = parser.get_text()
        except (httpx.HTTPStatusError, httpx.ConnectError):
            content = ""

        # Tier 2: Jina fallback — if tier 1 failed or returned junk
        if not _is_usable_content(content):
            try:
                content = await _fetch_via_jina(url)
            except Exception:
                pass  # keep whatever tier 1 returned

        if not _is_usable_content(content):
            return f"Failed to fetch {url}: no usable content (page may require JavaScript rendering)."

        if len(content) > FETCH_CHAR_LIMIT:
            content = content[:FETCH_CHAR_LIMIT] + f"\n\n[truncated — {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Failed to fetch {url}: {e}"

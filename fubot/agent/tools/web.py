"""Web tools: web_search and web_fetch."""

from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import os
import re
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

from fubot.agent.tools.base import Tool

if TYPE_CHECKING:
    from fubot.config.schema import WebSearchConfig

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks
MAX_FETCH_BYTES = 1_000_000
_BLOCKED_HOSTNAMES = {"localhost", "ip6-localhost", "ip6-loopback"}
_BLOCKED_HOST_SUFFIXES = (".local", ".localdomain", ".localhost", ".internal", ".lan", ".home", ".corp")


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


async def _validate_public_web_url(url: str) -> tuple[bool, str]:
    """Reject private, loopback, and local-only web targets."""
    is_valid, error_msg = _validate_url(url)
    if not is_valid:
        return False, error_msg

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").strip().rstrip(".").lower()
    if not hostname:
        return False, "Missing hostname"
    if hostname in _BLOCKED_HOSTNAMES or hostname.endswith(_BLOCKED_HOST_SUFFIXES):
        return False, f"Blocked local hostname '{hostname}'"

    try:
        target = ipaddress.ip_address(hostname)
        reason = _blocked_ip_reason(target)
        if reason:
            return False, reason
        return True, ""
    except ValueError:
        pass

    try:
        addresses = await _resolve_host_ips(hostname)
    except OSError as exc:
        return False, f"DNS resolution failed for '{hostname}': {exc}"

    if not addresses:
        return False, f"DNS resolution returned no addresses for '{hostname}'"

    for address in addresses:
        reason = _blocked_ip_reason(address)
        if reason:
            return False, reason
    return True, ""


def _blocked_ip_reason(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    if address.is_private:
        return f"Blocked private address '{address}'"
    if address.is_loopback:
        return f"Blocked loopback address '{address}'"
    if address.is_link_local:
        return f"Blocked link-local address '{address}'"
    if address.is_reserved:
        return f"Blocked reserved address '{address}'"
    if address.is_multicast:
        return f"Blocked multicast address '{address}'"
    if address.is_unspecified:
        return f"Blocked unspecified address '{address}'"
    return None


async def _resolve_host_ips(hostname: str) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    def _lookup() -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        resolved: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM):
            raw = sockaddr[0]
            if family == socket.AF_INET6 and "%" in raw:
                raw = raw.split("%", 1)[0]
            resolved.add(ipaddress.ip_address(raw))
        return resolved

    return await asyncio.to_thread(_lookup)


def _format_results(query: str, items: list[dict[str, Any]], n: int) -> str:
    """Format provider results into shared plaintext output."""
    if not items:
        return f"No results for: {query}"
    lines = [f"Results for: {query}\n"]
    for i, item in enumerate(items[:n], 1):
        title = _normalize(_strip_tags(item.get("title", "")))
        snippet = _normalize(_strip_tags(item.get("content", "")))
        lines.append(f"{i}. {title}\n   {item.get('url', '')}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


class WebSearchTool(Tool):
    """Search the web using configured provider."""

    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(self, config: WebSearchConfig | None = None, proxy: str | None = None):
        from fubot.config.schema import WebSearchConfig

        self.config = config if config is not None else WebSearchConfig()
        self.proxy = proxy

    def execution_mode(self, params: dict[str, Any]) -> str:
        _ = params
        return "read_only"

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        provider = self.config.provider.strip().lower() or "brave"
        n = min(max(count or self.config.max_results, 1), 10)

        if provider == "duckduckgo":
            return await self._search_duckduckgo(query, n)
        elif provider == "tavily":
            return await self._search_tavily(query, n)
        elif provider == "searxng":
            return await self._search_searxng(query, n)
        elif provider == "jina":
            return await self._search_jina(query, n)
        elif provider == "brave":
            return await self._search_brave(query, n)
        else:
            return f"Error: unknown search provider '{provider}'"

    async def _search_brave(self, query: str, n: int) -> str:
        api_key = self.config.api_key or os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            logger.warning("BRAVE_API_KEY not set, falling back to DuckDuckGo")
            return await self._search_duckduckgo(query, n)
        try:
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                    timeout=10.0,
                )
                r.raise_for_status()
            items = [
                {"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("description", "")}
                for x in r.json().get("web", {}).get("results", [])
            ]
            return _format_results(query, items, n)
        except Exception as e:
            return f"Error: {e}"

    async def _search_tavily(self, query: str, n: int) -> str:
        api_key = self.config.api_key or os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set, falling back to DuckDuckGo")
            return await self._search_duckduckgo(query, n)
        try:
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"query": query, "max_results": n},
                    timeout=15.0,
                )
                r.raise_for_status()
            return _format_results(query, r.json().get("results", []), n)
        except Exception as e:
            return f"Error: {e}"

    async def _search_searxng(self, query: str, n: int) -> str:
        base_url = (self.config.base_url or os.environ.get("SEARXNG_BASE_URL", "")).strip()
        if not base_url:
            logger.warning("SEARXNG_BASE_URL not set, falling back to DuckDuckGo")
            return await self._search_duckduckgo(query, n)
        endpoint = f"{base_url.rstrip('/')}/search"
        is_valid, error_msg = _validate_url(endpoint)
        if not is_valid:
            return f"Error: invalid SearXNG URL: {error_msg}"
        try:
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.get(
                    endpoint,
                    params={"q": query, "format": "json"},
                    headers={"User-Agent": USER_AGENT},
                    timeout=10.0,
                )
                r.raise_for_status()
            return _format_results(query, r.json().get("results", []), n)
        except Exception as e:
            return f"Error: {e}"

    async def _search_jina(self, query: str, n: int) -> str:
        api_key = self.config.api_key or os.environ.get("JINA_API_KEY", "")
        if not api_key:
            logger.warning("JINA_API_KEY not set, falling back to DuckDuckGo")
            return await self._search_duckduckgo(query, n)
        try:
            headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.get(
                    "https://s.jina.ai/",
                    params={"q": query},
                    headers=headers,
                    timeout=15.0,
                )
                r.raise_for_status()
            data = r.json().get("data", [])[:n]
            items = [
                {"title": d.get("title", ""), "url": d.get("url", ""), "content": d.get("content", "")[:500]}
                for d in data
            ]
            return _format_results(query, items, n)
        except Exception as e:
            return f"Error: {e}"

    async def _search_duckduckgo(self, query: str, n: int) -> str:
        try:
            from ddgs import DDGS

            ddgs = DDGS(timeout=10)
            raw = await asyncio.to_thread(ddgs.text, query, max_results=n)
            if not raw:
                return f"No results for: {query}"
            items = [
                {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
                for r in raw
            ]
            return _format_results(query, items, n)
        except Exception as e:
            logger.warning("DuckDuckGo search failed: {}", e)
            return f"Error: DuckDuckGo search failed ({e})"


class WebFetchTool(Tool):
    """Fetch and extract content from a URL."""

    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100},
        },
        "required": ["url"],
    }

    def __init__(self, max_chars: int = 50000, proxy: str | None = None, max_response_bytes: int = MAX_FETCH_BYTES):
        self.max_chars = max_chars
        self.proxy = proxy
        self.max_response_bytes = max_response_bytes

    def execution_mode(self, params: dict[str, Any]) -> str:
        _ = params
        return "read_only"

    async def execute(
        self,
        url: str,
        extract_mode: str = "markdown",
        max_chars_value: int | None = None,
        **kwargs: Any,
    ) -> str:
        extract_mode = kwargs.get("extractMode", extract_mode)
        max_chars = kwargs.get("maxChars", max_chars_value) or self.max_chars
        is_valid, error_msg = await _validate_public_web_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url}, ensure_ascii=False)

        result = await self._fetch_jina(url, max_chars)
        if result is None:
            result = await self._fetch_readability(url, extract_mode, max_chars)
        return result

    async def _fetch_jina(self, url: str, max_chars: int) -> str | None:
        """Try fetching via Jina Reader API. Returns None on failure."""
        try:
            headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
            jina_key = os.environ.get("JINA_API_KEY", "")
            if jina_key:
                headers["Authorization"] = f"Bearer {jina_key}"
            async with httpx.AsyncClient(proxy=self.proxy, timeout=20.0) as client:
                async with client.stream("GET", f"https://r.jina.ai/{url}", headers=headers) as response:
                    if response.status_code == 429:
                        logger.debug("Jina Reader rate limited, falling back to readability")
                        return None
                    response.raise_for_status()
                    payload = await self._read_response_bytes(response)
                if response.status_code == 429:
                    logger.debug("Jina Reader rate limited, falling back to readability")
                    return None
            data = json.loads(payload.decode("utf-8", errors="replace")).get("data", {})
            title = data.get("title", "")
            text = data.get("content", "")
            if not text:
                return None
            final_url = data.get("url", url)
            is_valid, error_msg = await _validate_public_web_url(final_url)
            if not is_valid:
                return json.dumps({"error": f"URL validation failed: {error_msg}", "url": final_url}, ensure_ascii=False)

            if title:
                text = f"# {title}\n\n{text}"
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps({
                "url": url, "finalUrl": final_url, "status": response.status_code,
                "extractor": "jina", "truncated": truncated, "length": len(text), "text": text,
            }, ensure_ascii=False)
        except Exception as e:
            logger.debug("Jina Reader failed for {}, falling back to readability: {}", url, e)
            return None

    async def _fetch_readability(self, url: str, extract_mode: str, max_chars: int) -> str:
        """Local fallback using readability-lxml."""
        from readability import Document

        try:
            response_meta, body = await self._fetch_with_redirect_validation(url)
            ctype = response_meta["headers"].get("content-type", "")
            text_body = body.decode(response_meta["encoding"] or "utf-8", errors="replace")

            if "application/json" in ctype:
                text, extractor = json.dumps(json.loads(text_body), indent=2, ensure_ascii=False), "json"
            elif "text/html" in ctype or text_body[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(text_body)
                content = self._to_markdown(doc.summary()) if extract_mode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = text_body, "raw"

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps({
                "url": url, "finalUrl": response_meta["final_url"], "status": response_meta["status_code"],
                "extractor": extractor, "truncated": truncated, "length": len(text), "text": text,
            }, ensure_ascii=False)
        except httpx.ProxyError as e:
            logger.error("WebFetch proxy error for {}: {}", url, e)
            return json.dumps({"error": f"Proxy error: {e}", "url": url}, ensure_ascii=False)
        except ValueError as e:
            logger.warning("WebFetch blocked {}: {}", url, e)
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)
        except Exception as e:
            logger.error("WebFetch error for {}: {}", url, e)
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)

    async def _fetch_with_redirect_validation(self, url: str) -> tuple[dict[str, Any], bytes]:
        headers = {"User-Agent": USER_AGENT}
        current_url = url
        async with httpx.AsyncClient(
            follow_redirects=False,
            max_redirects=MAX_REDIRECTS,
            timeout=30.0,
            proxy=self.proxy,
        ) as client:
            for redirect_index in range(MAX_REDIRECTS + 1):
                is_valid, error_msg = await _validate_public_web_url(current_url)
                if not is_valid:
                    raise ValueError(f"URL validation failed: {error_msg}")

                async with client.stream("GET", current_url, headers=headers) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            response.raise_for_status()
                        current_url = urljoin(str(response.url), location)
                        continue

                    response.raise_for_status()
                    payload = await self._read_response_bytes(response)
                    return (
                        {
                            "status_code": response.status_code,
                            "final_url": str(response.url),
                            "headers": response.headers,
                            "encoding": response.encoding,
                        },
                        payload,
                    )

        raise ValueError(f"Too many redirects while fetching {url}")

    async def _read_response_bytes(self, response: httpx.Response) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                parsed_length = int(content_length)
            except ValueError:
                pass
            else:
                if parsed_length > self.max_response_bytes:
                    raise ValueError(
                        f"Response body exceeds limit ({self.max_response_bytes} bytes)",
                    )

        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.max_response_bytes:
                raise ValueError(f"Response body exceeds limit ({self.max_response_bytes} bytes)")
            chunks.append(chunk)
        return b"".join(chunks)

    def _to_markdown(self, html_content: str) -> str:
        """Convert HTML to markdown."""
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html_content, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))

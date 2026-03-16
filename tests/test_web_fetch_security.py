from __future__ import annotations

import json
from ipaddress import ip_address
from unittest.mock import AsyncMock

import httpx
import pytest

from fubot.agent.tools.web import WebFetchTool

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_client(handler):
    transport = httpx.MockTransport(handler)

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    return _factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://10.0.0.5",
        "http://172.16.0.9",
        "http://192.168.1.2",
        "http://[::1]/",
    ],
)
async def test_web_fetch_blocks_private_and_loopback_targets(url: str) -> None:
    tool = WebFetchTool()

    result = await tool.execute(url)
    payload = json.loads(result)

    assert "URL validation failed" in payload["error"]


@pytest.mark.asyncio
async def test_web_fetch_blocks_redirect_to_private_address(monkeypatch) -> None:
    tool = WebFetchTool()
    monkeypatch.setattr(tool, "_fetch_jina", AsyncMock(return_value=None))

    async def _resolve(hostname: str):
        mapping = {
            "example.com": {ip_address("93.184.216.34")},
            "127.0.0.1": {ip_address("127.0.0.1")},
        }
        return mapping[hostname]

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/internal"},
                request=request,
            )
        raise AssertionError(f"unexpected redirect target {request.url!s}")

    monkeypatch.setattr("fubot.agent.tools.web._resolve_host_ips", _resolve)
    monkeypatch.setattr("fubot.agent.tools.web.httpx.AsyncClient", _mock_client(_handler))

    result = await tool.execute("http://example.com/start")
    payload = json.loads(result)

    assert "URL validation failed" in payload["error"]
    assert "127.0.0.1" in payload["error"]


@pytest.mark.asyncio
async def test_web_fetch_rejects_oversized_response_bodies(monkeypatch) -> None:
    tool = WebFetchTool(max_response_bytes=16)
    monkeypatch.setattr(tool, "_fetch_jina", AsyncMock(return_value=None))

    async def _resolve(hostname: str):
        return {ip_address("93.184.216.34")}

    body = b"x" * 32

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-length": str(len(body))},
            content=body,
            request=request,
        )

    monkeypatch.setattr("fubot.agent.tools.web._resolve_host_ips", _resolve)
    monkeypatch.setattr("fubot.agent.tools.web.httpx.AsyncClient", _mock_client(_handler))

    result = await tool.execute("http://example.com/large")
    payload = json.loads(result)

    assert "exceeds limit" in payload["error"]

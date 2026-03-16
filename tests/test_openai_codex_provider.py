from __future__ import annotations

from types import SimpleNamespace

import pytest

from fubot.providers.openai_codex_provider import OpenAICodexProvider


@pytest.mark.asyncio
async def test_codex_provider_does_not_retry_with_insecure_tls(monkeypatch) -> None:
    provider = OpenAICodexProvider(default_model="openai-codex/gpt-5.1-codex")
    calls: list[tuple[str, dict, dict]] = []

    monkeypatch.setattr(
        "fubot.providers.openai_codex_provider.get_codex_token",
        lambda: SimpleNamespace(account_id="acct", access="token"),
    )

    async def _request(url, headers, body):
        calls.append((url, headers, body))
        raise RuntimeError("CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr("fubot.providers.openai_codex_provider._request_codex", _request)

    response = await provider.chat(messages=[{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert "CERTIFICATE_VERIFY_FAILED" in (response.content or "")
    assert len(calls) == 1

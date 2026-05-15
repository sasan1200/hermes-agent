"""Tests for the optional Discord Interactions HTTPS gateway adapter."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nacl.signing import SigningKey

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageType
from gateway.platforms.discord_interactions import (
    DISCORD_APPLICATION_COMMAND,
    DISCORD_DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE,
    DISCORD_PING,
    DISCORD_PONG,
    DiscordInteractionsAdapter,
    check_discord_interactions_requirements,
    verify_discord_signature,
)
from gateway.run import GatewayRunner


class _FakeRequest:
    def __init__(self, raw_body: bytes, headers: dict[str, str]):
        self._raw_body = raw_body
        self.headers = headers
        self.content_length = len(raw_body)

    async def read(self) -> bytes:
        return self._raw_body


def _signed_request(payload: dict, signing_key: SigningKey) -> tuple[_FakeRequest, str]:
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = "1778860000"
    signature = signing_key.sign(timestamp.encode("utf-8") + raw_body).signature.hex()
    request = _FakeRequest(
        raw_body,
        {
            "X-Signature-Ed25519": signature,
            "X-Signature-Timestamp": timestamp,
        },
    )
    return request, signing_key.verify_key.encode().hex()


def _json_response_body(response) -> dict:
    return json.loads(response.text)


def _adapter(**extra) -> DiscordInteractionsAdapter:
    config = PlatformConfig(
        enabled=True,
        extra={
            "public_key": "00" * 32,
            "application_id": "app-123",
            **extra,
        },
    )
    return DiscordInteractionsAdapter(config)


def test_signature_verification_uses_timestamp_plus_exact_raw_body():
    signing_key = SigningKey.generate()
    raw_body = b'{"type":1}'
    timestamp = "1778860000"
    signature = signing_key.sign(timestamp.encode("utf-8") + raw_body).signature.hex()
    public_key = signing_key.verify_key.encode().hex()

    assert verify_discord_signature(public_key, timestamp, raw_body, signature) is True
    assert verify_discord_signature(public_key, timestamp, b'{"type":2}', signature) is False
    assert verify_discord_signature(public_key, "1778860001", raw_body, signature) is False


def test_requirements_report_aiohttp_and_pynacl_available():
    assert check_discord_interactions_requirements() is True


@pytest.mark.asyncio
async def test_ping_returns_pong_after_valid_ed25519_signature():
    signing_key = SigningKey.generate()
    request, public_key = _signed_request({"type": DISCORD_PING}, signing_key)
    adapter = _adapter(public_key=public_key)

    response = await adapter._handle_interaction(request)

    assert response.status == 200
    assert _json_response_body(response) == {"type": DISCORD_PONG}


@pytest.mark.asyncio
async def test_tampered_body_is_rejected_before_json_processing():
    signing_key = SigningKey.generate()
    request, public_key = _signed_request({"type": DISCORD_PING}, signing_key)
    request._raw_body = b'{"type":2}'
    request.content_length = len(request._raw_body)
    adapter = _adapter(public_key=public_key)

    response = await adapter._handle_interaction(request)

    assert response.status == 401
    assert "Invalid signature" in response.text


@pytest.mark.asyncio
async def test_missing_signature_headers_are_rejected():
    adapter = _adapter(public_key=SigningKey.generate().verify_key.encode().hex())

    response = await adapter._handle_interaction(_FakeRequest(b'{"type":1}', {}))

    assert response.status == 401


@pytest.mark.asyncio
async def test_application_command_defers_and_dispatches_background_message_event():
    signing_key = SigningKey.generate()
    payload = {
        "id": "inter-123",
        "application_id": "app-123",
        "token": "tok_abcdef",
        "type": DISCORD_APPLICATION_COMMAND,
        "guild_id": "guild-1",
        "channel_id": "chan-1",
        "member": {
            "user": {"id": "user-1", "username": "franco"},
            "roles": ["role-1"],
        },
        "data": {"name": "ask", "options": [{"name": "prompt", "value": "hello"}]},
    }
    request, public_key = _signed_request(payload, signing_key)
    adapter = _adapter(public_key=public_key)
    adapter.handle_message = AsyncMock()

    response = await adapter._handle_interaction(request)
    await asyncio.sleep(0)

    assert response.status == 200
    assert _json_response_body(response) == {"type": DISCORD_DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE}
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "/ask --prompt hello"
    assert event.message_type == MessageType.TEXT
    assert event.message_id == "inter-123"
    assert event.raw_message == payload
    assert event.source.platform == Platform.DISCORD_INTERACTIONS
    assert event.source.chat_id == "discord_interactions:chan-1:inter-123"
    assert event.source.chat_type == "channel"
    assert event.source.user_id == "user-1"
    assert event.source.user_name == "franco"
    assert event.source.guild_id == "guild-1"
    assert adapter._interaction_context[event.source.chat_id]["token"] == "tok_abcdef"


@pytest.mark.asyncio
async def test_user_allowlist_allows_matching_user_and_rejects_others():
    signing_key = SigningKey.generate()
    payload = {
        "id": "inter-123",
        "application_id": "app-123",
        "token": "tok",
        "type": DISCORD_APPLICATION_COMMAND,
        "channel_id": "chan-1",
        "member": {"user": {"id": "user-1", "username": "allowed"}},
        "data": {"name": "ask"},
    }
    request, public_key = _signed_request(payload, signing_key)
    adapter = _adapter(public_key=public_key, allowed_users=["user-1"])
    adapter.handle_message = AsyncMock()

    assert (await adapter._handle_interaction(request)).status == 200

    payload["id"] = "inter-456"
    payload["member"]["user"]["id"] = "user-2"
    request, _ = _signed_request(payload, signing_key)
    response = await adapter._handle_interaction(request)

    assert response.status == 403
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_role_allowlist_is_scoped_to_originating_guild_member_roles():
    signing_key = SigningKey.generate()
    payload = {
        "id": "inter-123",
        "application_id": "app-123",
        "token": "tok",
        "type": DISCORD_APPLICATION_COMMAND,
        "guild_id": "guild-1",
        "channel_id": "chan-1",
        "member": {
            "user": {"id": "user-1", "username": "role-user"},
            "roles": ["role-1"],
        },
        "data": {"name": "ask"},
    }
    request, public_key = _signed_request(payload, signing_key)
    adapter = _adapter(public_key=public_key, allowed_roles=["role-1"])
    adapter.handle_message = AsyncMock()

    assert (await adapter._handle_interaction(request)).status == 200

    payload["id"] = "inter-456"
    payload["member"]["roles"] = ["other-guild-role"]
    request, _ = _signed_request(payload, signing_key)
    response = await adapter._handle_interaction(request)

    assert response.status == 403
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_dm_does_not_authorize_roles_without_guild_member_payload():
    signing_key = SigningKey.generate()
    payload = {
        "id": "inter-123",
        "application_id": "app-123",
        "token": "tok",
        "type": DISCORD_APPLICATION_COMMAND,
        "channel_id": "dm-chan",
        "user": {"id": "user-1", "username": "dm-user"},
        "data": {"name": "ask"},
    }
    request, public_key = _signed_request(payload, signing_key)
    adapter = _adapter(public_key=public_key, allowed_roles=["role-1"])
    adapter.handle_message = AsyncMock()

    response = await adapter._handle_interaction(request)

    assert response.status == 403
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_uses_interaction_followup_webhook(monkeypatch):
    adapter = _adapter(application_id="app-123")
    chat_id = "discord_interactions:chan-1:inter-123"
    adapter._interaction_context[chat_id] = {"application_id": "app-123", "token": "tok-xyz"}
    posts = []

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {"id": "msg-1"}

        async def text(self):
            return "ok"

    class _Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, **kwargs):
            posts.append((url, kwargs))
            return _Response()

    monkeypatch.setattr("gateway.platforms.discord_interactions.aiohttp.ClientSession", _Session)

    result = await adapter.send(chat_id, "hello from hermes")

    assert result.success is True
    assert result.message_id == "msg-1"
    assert posts == [
        (
            "https://discord.com/api/v10/webhooks/app-123/tok-xyz",
            {"json": {"content": "hello from hermes"}},
        )
    ]


def test_gateway_factory_registers_discord_interactions_adapter():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(group_sessions_per_user=False, thread_sessions_per_user=False)
    config = PlatformConfig(enabled=True, extra={"public_key": "00" * 32, "application_id": "app-123"})

    adapter = runner._create_adapter(Platform.DISCORD_INTERACTIONS, config)

    assert isinstance(adapter, DiscordInteractionsAdapter)


def test_env_overrides_enable_discord_interactions_platform(monkeypatch):
    monkeypatch.setenv("DISCORD_INTERACTIONS_PUBLIC_KEY", "aa" * 32)
    monkeypatch.setenv("DISCORD_INTERACTIONS_APPLICATION_ID", "app-123")
    monkeypatch.setenv("DISCORD_INTERACTIONS_HOST", "0.0.0.0")
    monkeypatch.setenv("DISCORD_INTERACTIONS_PORT", "9876")
    monkeypatch.setenv("DISCORD_INTERACTIONS_PATH", "/custom/interactions")
    config = GatewayConfig()

    _apply_env_overrides(config)

    platform_config = config.platforms[Platform.DISCORD_INTERACTIONS]
    assert platform_config.enabled is True
    assert platform_config.extra == {
        "public_key": "aa" * 32,
        "application_id": "app-123",
        "host": "0.0.0.0",
        "port": 9876,
        "path": "/custom/interactions",
    }

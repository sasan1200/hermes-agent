"""Discord Interactions HTTPS platform adapter.

This optional adapter receives Discord application-command interactions over an
HTTPS endpoint instead of the long-lived discord.py gateway websocket. It keeps
that protocol surface separate from :mod:`gateway.platforms.discord` while
reusing Hermes' platform abstractions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

try:
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by requirement checks
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    PYNACL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by requirement checks
    BadSignatureError = Exception  # type: ignore[assignment]
    VerifyKey = None  # type: ignore[assignment]
    PYNACL_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8645
DEFAULT_PATH = "/discord/interactions"
DISCORD_API_BASE = "https://discord.com/api/v10"

DISCORD_PING = 1
DISCORD_APPLICATION_COMMAND = 2
DISCORD_PONG = 1
DISCORD_DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE = 5

_SIGNATURE_HEADER = "X-Signature-Ed25519"
_TIMESTAMP_HEADER = "X-Signature-Timestamp"


def check_discord_interactions_requirements() -> bool:
    """Return True when dependencies for Discord Interactions are importable."""
    return AIOHTTP_AVAILABLE and PYNACL_AVAILABLE


def verify_discord_signature(
    public_key_hex: str,
    timestamp: str,
    raw_body: bytes,
    signature_hex: str,
) -> bool:
    """Validate Discord's Ed25519 request signature.

    Discord signs ``timestamp + raw_request_body``. The raw bytes must be used
    exactly as received and validation must happen before JSON parsing.
    """
    if not PYNACL_AVAILABLE or VerifyKey is None:
        return False
    if not public_key_hex or not timestamp or not signature_hex:
        return False
    try:
        key = VerifyKey(bytes.fromhex(str(public_key_hex).strip()))
        key.verify(
            timestamp.encode("utf-8") + raw_body,
            bytes.fromhex(str(signature_hex).strip()),
        )
        return True
    except (ValueError, BadSignatureError):
        return False


def _list_from_config_or_env(value: Any, env_var: str) -> set[str]:
    """Normalize a list-like config value or comma-separated env fallback."""
    if value is None or value == "":
        value = os.getenv(env_var, "")
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return {str(item).strip() for item in items if str(item).strip()}


def _bool_from_config_or_env(value: Any, env_var: str, default: bool = False) -> bool:
    if value is None:
        value = os.getenv(env_var, "")
    if value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class DiscordInteractionsAdapter(BasePlatformAdapter):
    """Receive Discord slash commands via the Interactions HTTPS protocol."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD_INTERACTIONS)
        extra = config.extra or {}
        self._host = str(extra.get("host") or os.getenv("DISCORD_INTERACTIONS_HOST") or DEFAULT_HOST)
        self._port = int(extra.get("port") or os.getenv("DISCORD_INTERACTIONS_PORT") or DEFAULT_PORT)
        raw_path = str(extra.get("path") or os.getenv("DISCORD_INTERACTIONS_PATH") or DEFAULT_PATH)
        self._path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
        self._public_key = str(
            extra.get("public_key")
            or os.getenv("DISCORD_INTERACTIONS_PUBLIC_KEY")
            or ""
        ).strip()
        self._application_id = str(
            extra.get("application_id")
            or os.getenv("DISCORD_INTERACTIONS_APPLICATION_ID")
            or ""
        ).strip()
        self._max_body_bytes = int(extra.get("max_body_bytes", 128 * 1024))
        self._allowed_user_ids = _list_from_config_or_env(
            extra.get("allowed_users"), "DISCORD_ALLOWED_USERS"
        )
        self._allowed_role_ids = _list_from_config_or_env(
            extra.get("allowed_roles"), "DISCORD_ALLOWED_ROLES"
        )
        self._allowed_channel_ids = _list_from_config_or_env(
            extra.get("allowed_channels"), "DISCORD_ALLOWED_CHANNELS"
        )
        self._ignored_channel_ids = _list_from_config_or_env(
            extra.get("ignored_channels"), "DISCORD_IGNORED_CHANNELS"
        )
        self._dm_role_auth_enabled = _bool_from_config_or_env(
            extra.get("dm_role_auth"), "DISCORD_INTERACTIONS_DM_ROLE_AUTH", False
        )
        self._runner: Optional["web.AppRunner"] = None
        self._interaction_context: Dict[str, dict[str, Any]] = {}

    async def connect(self) -> bool:
        if not check_discord_interactions_requirements():
            logger.warning("DiscordInteractions: aiohttp/PyNaCl not installed")
            return False
        if not self._public_key:
            raise ValueError(
                "[discord_interactions] public_key is required. Set "
                "platforms.discord_interactions.extra.public_key or "
                "DISCORD_INTERACTIONS_PUBLIC_KEY."
            )

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._path, self._handle_interaction)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info(
            "[discord_interactions] Listening on %s:%d%s",
            self._host,
            self._port,
            self._path,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[discord_interactions] Disconnected")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "discord_interactions"}

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send agent output through Discord's interaction follow-up webhook."""
        context = dict(self._interaction_context.get(chat_id) or {})
        if metadata:
            context.update(metadata.get("discord_interactions", {}) or {})
        application_id = str(context.get("application_id") or self._application_id or "")
        token = str(context.get("token") or "")
        if not application_id or not token:
            return SendResult(
                success=False,
                error="Missing Discord interaction application_id/token for follow-up",
            )

        url = f"{DISCORD_API_BASE}/webhooks/{application_id}/{token}"
        payload = {"content": content}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    body_text = await response.text()
                    if response.status >= 400:
                        return SendResult(
                            success=False,
                            error=f"Discord follow-up failed: HTTP {response.status}: {body_text[:200]}",
                        )
                    try:
                        data = await response.json()
                    except Exception:
                        data = {"text": body_text}
                    return SendResult(
                        success=True,
                        message_id=str(data.get("id")) if isinstance(data, dict) and data.get("id") else None,
                        raw_response=data,
                    )
        except Exception as exc:
            logger.warning("[discord_interactions] follow-up send failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "discord_interactions"})

    async def _handle_interaction(self, request: "web.Request") -> "web.Response":
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"error": "Payload too large"}, status=413)

        try:
            raw_body = await request.read()
        except Exception:
            return web.json_response({"error": "Bad request"}, status=400)

        timestamp = request.headers.get(_TIMESTAMP_HEADER, "")
        signature = request.headers.get(_SIGNATURE_HEADER, "")
        if not verify_discord_signature(self._public_key, timestamp, raw_body, signature):
            logger.warning("[discord_interactions] Invalid Discord signature")
            return web.json_response({"error": "Invalid signature"}, status=401)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        interaction_type = payload.get("type")
        if interaction_type == DISCORD_PING:
            return web.json_response({"type": DISCORD_PONG})

        if interaction_type != DISCORD_APPLICATION_COMMAND:
            return web.json_response({"error": "Unsupported interaction type"}, status=400)

        allowed, reason = self._is_payload_authorized(payload)
        if not allowed:
            logger.warning("[discord_interactions] Unauthorized interaction rejected: %s", reason)
            return web.json_response({"error": "Unauthorized"}, status=403)

        event = self._payload_to_event(payload)
        self._interaction_context[event.source.chat_id] = {
            "application_id": str(payload.get("application_id") or self._application_id or ""),
            "token": str(payload.get("token") or ""),
            "interaction_id": str(payload.get("id") or ""),
        }

        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        # Yield once so lightweight/mocked handlers can start without delaying
        # Discord's deferred ACK path for real agent work.
        await asyncio.sleep(0)
        return web.json_response({"type": DISCORD_DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE})

    def _is_payload_authorized(self, payload: dict[str, Any]) -> tuple[bool, str | None]:
        channel_id = str(payload.get("channel_id") or "")
        guild_id = str(payload.get("guild_id") or "")
        is_dm = not bool(guild_id)

        channel_ids = {channel_id} if channel_id else set()
        if self._allowed_channel_ids and "*" not in self._allowed_channel_ids:
            if not channel_ids or not (channel_ids & self._allowed_channel_ids):
                return False, "channel not in DISCORD_ALLOWED_CHANNELS"
        if self._ignored_channel_ids and ("*" in self._ignored_channel_ids or channel_ids & self._ignored_channel_ids):
            return False, "channel in DISCORD_IGNORED_CHANNELS"

        user = self._payload_user(payload)
        has_user_policy = bool(self._allowed_user_ids)
        has_role_policy = bool(self._allowed_role_ids)
        if not has_user_policy and not has_role_policy:
            return True, None
        user_id = str(user.get("id") or "") if user else ""
        if not user_id:
            return False, "missing user with allowlist configured"
        if has_user_policy and user_id in self._allowed_user_ids:
            return True, None
        if not has_role_policy:
            return False, "user not in DISCORD_ALLOWED_USERS"
        if is_dm and not self._dm_role_auth_enabled:
            return False, "DM role authorization disabled"
        member = payload.get("member") if isinstance(payload.get("member"), dict) else {}
        role_ids = {str(role).strip() for role in member.get("roles") or [] if str(role).strip()}
        if role_ids & self._allowed_role_ids:
            return True, None
        return False, "user not in DISCORD_ALLOWED_USERS / DISCORD_ALLOWED_ROLES"

    def _payload_to_event(self, payload: dict[str, Any]) -> MessageEvent:
        interaction_id = str(payload.get("id") or "")
        channel_id = str(payload.get("channel_id") or payload.get("guild_id") or "dm")
        guild_id = str(payload.get("guild_id") or "")
        chat_id = f"discord_interactions:{channel_id}:{interaction_id}"
        user = self._payload_user(payload)
        user_id = str(user.get("id") or "unknown") if user else "unknown"
        user_name = str(
            user.get("username")
            or user.get("global_name")
            or user.get("name")
            or "discord-user"
        ) if user else "discord-user"
        source = self.build_source(
            chat_id=chat_id,
            chat_name=f"discord/{channel_id}",
            chat_type="channel" if guild_id else "dm",
            user_id=user_id,
            user_name=user_name,
            guild_id=guild_id or None,
            message_id=interaction_id,
        )
        return MessageEvent(
            text=self._command_text(payload),
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=interaction_id,
        )

    @staticmethod
    def _payload_user(payload: dict[str, Any]) -> dict[str, Any]:
        member = payload.get("member") if isinstance(payload.get("member"), dict) else {}
        user = member.get("user") or payload.get("user") or {}
        return user if isinstance(user, dict) else {}

    @staticmethod
    def _command_text(payload: dict[str, Any]) -> str:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        command = str(data.get("name") or "").strip()
        if not command:
            return ""
        pairs: list[str] = []
        for opt in data.get("options") or []:
            if not isinstance(opt, dict):
                continue
            name = str(opt.get("name") or "").strip()
            if not name or "value" not in opt:
                continue
            pairs.append(f"--{name} {opt.get('value')}")
        args = " ".join(pairs)
        return f"/{command}" + (f" {args}" if args else "")

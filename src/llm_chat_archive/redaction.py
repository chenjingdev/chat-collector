from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

REDACTED_VALUE = "[REDACTED]"
REDACTED_API_KEY = "[REDACTED_API_KEY]"

_TOKEN_FIELD_PATTERN = (
    r"(?:access(?:_|-)?token|refresh(?:_|-)?token|id(?:_|-)?token|"
    r"client(?:_|-)?secret|api(?:_|-)?key|authorization|auth(?:_|-)?token|"
    r"oauth(?:_|-)?token|token)"
)
_ASSIGNMENT_FIELD_PATTERN = (
    r"(?:access(?:_|-)?token|refresh(?:_|-)?token|id(?:_|-)?token|"
    r"client(?:_|-)?secret|api(?:_|-)?key|auth(?:_|-)?token|oauth(?:_|-)?token|token)"
)
_SECRET_FIELD_NAMES = frozenset(
    {
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "clientsecret",
        "apikey",
        "authorization",
        "authtoken",
        "oauthtoken",
        "token",
        "bearertoken",
        "sessiontoken",
    }
)
_AUTHORIZATION_BEARER_PATTERN = re.compile(
    r"(?i)\b(Authorization\s*:\s*Bearer\s+)([A-Za-z0-9._~+/=-]+)"
)
_DOUBLE_QUOTED_SECRET_FIELD_PATTERN = re.compile(
    rf'(?i)("({_TOKEN_FIELD_PATTERN})"\s*:\s*")[^"]*(")'
)
_SINGLE_QUOTED_SECRET_FIELD_PATTERN = re.compile(
    rf"(?i)('({_TOKEN_FIELD_PATTERN})'\s*:\s*')[^']*(')"
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    rf"(?i)\b(({_ASSIGNMENT_FIELD_PATTERN})\s*[=:]\s*)([^\s,;]+)"
)
_OPENAI_API_KEY_PATTERN = re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{16,}\b")
_ANTHROPIC_API_KEY_PATTERN = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b")
_GOOGLE_API_KEY_PATTERN = re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b")


@dataclass(frozen=True, slots=True)
class RedactionResult:
    payload: dict[str, object]
    event_count: int


def redact_archive_payload(payload: Mapping[str, object]) -> RedactionResult:
    redacted_payload = dict(payload)
    event_count = 0

    messages = payload.get("messages")
    if isinstance(messages, list):
        redacted_messages: list[object] = []
        for message in messages:
            if not isinstance(message, dict):
                redacted_messages.append(message)
                continue

            redacted_message = dict(message)
            text = message.get("text")
            if isinstance(text, str):
                redacted_text, redacted_count = redact_text(text)
                redacted_message["text"] = redacted_text
                event_count += redacted_count
            redacted_messages.append(redacted_message)
        redacted_payload["messages"] = redacted_messages

    for key in ("session_metadata", "provenance"):
        value = payload.get(key)
        if value is None:
            continue
        redacted_value, redacted_count = _redact_metadata_value(value)
        redacted_payload[key] = redacted_value
        event_count += redacted_count

    return RedactionResult(payload=redacted_payload, event_count=event_count)


def redact_text(text: str) -> tuple[str, int]:
    redacted = text
    event_count = 0

    redacted, count = _AUTHORIZATION_BEARER_PATTERN.subn(r"\1[REDACTED]", redacted)
    event_count += count

    redacted, count = _DOUBLE_QUOTED_SECRET_FIELD_PATTERN.subn(r"\1[REDACTED]\3", redacted)
    event_count += count

    redacted, count = _SINGLE_QUOTED_SECRET_FIELD_PATTERN.subn(r"\1[REDACTED]\3", redacted)
    event_count += count

    redacted, count = _SECRET_ASSIGNMENT_PATTERN.subn(r"\1[REDACTED]", redacted)
    event_count += count

    redacted, count = _OPENAI_API_KEY_PATTERN.subn(REDACTED_API_KEY, redacted)
    event_count += count

    redacted, count = _ANTHROPIC_API_KEY_PATTERN.subn(REDACTED_API_KEY, redacted)
    event_count += count

    redacted, count = _GOOGLE_API_KEY_PATTERN.subn(REDACTED_API_KEY, redacted)
    event_count += count

    return redacted, event_count


def _redact_metadata_value(
    value: object,
    *,
    field_name: str | None = None,
) -> tuple[object, int]:
    if isinstance(value, str):
        if field_name is not None and _is_secret_field_name(field_name):
            if value.startswith("[REDACTED"):
                return value, 0
            return REDACTED_VALUE, 1
        return redact_text(value)

    if isinstance(value, dict):
        redacted: dict[object, object] = {}
        event_count = 0
        for key, item in value.items():
            nested_field_name = key if isinstance(key, str) else None
            redacted_item, nested_count = _redact_metadata_value(
                item,
                field_name=nested_field_name,
            )
            redacted[key] = redacted_item
            event_count += nested_count
        return redacted, event_count

    if isinstance(value, list):
        redacted_items: list[object] = []
        event_count = 0
        for item in value:
            redacted_item, nested_count = _redact_metadata_value(
                item,
                field_name=field_name,
            )
            redacted_items.append(redacted_item)
            event_count += nested_count
        return redacted_items, event_count

    return value, 0


def _is_secret_field_name(field_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", field_name.lower())
    return normalized in _SECRET_FIELD_NAMES


__all__ = [
    "REDACTED_API_KEY",
    "REDACTED_VALUE",
    "RedactionResult",
    "redact_archive_payload",
    "redact_text",
]

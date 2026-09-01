"""Content-blind transport resilience primitives for future P9 provider calls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import stat
from typing import Mapping, Sequence

from .p9_controller import P9ControllerError
from .p9e_contract import canonical_json, sha256_json


DEFAULT_RETRY_DELAYS_SECONDS = (2, 4, 8, 16, 30)
SUPPORTED_RETRYABLE_HTTP = frozenset({500, 502, 503, 504})
SAFE_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_sequence",
        "logical_request_sequence",
        "retry_ordinal",
        "next_retry_ordinal",
        "next_retry_delay_seconds",
        "request_sha256",
        "request_bytes",
        "model_sha256",
        "message_count",
        "message_role_counts",
        "message_bytes",
        "tool_count",
        "tool_bytes",
        "request_content_recorded",
        "credential_recorded",
        "response_sha256",
        "response_bytes",
        "status",
        "status_class",
        "request_id",
        "provider_response_received",
        "elapsed_milliseconds",
        "response_content_recorded",
        "error_class",
    }
)


@dataclass(frozen=True)
class TransportResiliencePolicy:
    """Frozen transport-only policy; it cannot alter prompts or tool schemas."""

    retryable_http_statuses: frozenset[int] = SUPPORTED_RETRYABLE_HTTP
    retry_delays_seconds: tuple[int, ...] = DEFAULT_RETRY_DELAYS_SECONDS
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.retryable_http_statuses != SUPPORTED_RETRYABLE_HTTP:
            raise P9ControllerError("P9 resilience retryable HTTP set drifted")
        if (
            len(self.retry_delays_seconds) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.retry_delays_seconds
            )
            or sum(self.retry_delays_seconds) > 60
        ):
            raise P9ControllerError("P9 resilience retry delay schedule drifted")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds != 300
        ):
            raise P9ControllerError("P9 resilience timeout drifted")

    @property
    def maximum_attempts(self) -> int:
        return len(self.retry_delays_seconds) + 1

    def as_record(self) -> dict[str, object]:
        return {
            "format": "dosee.p9-transport-resilience-policy.v1",
            "retryable_http_statuses": sorted(self.retryable_http_statuses),
            "retry_delays_seconds": list(self.retry_delays_seconds),
            "maximum_attempts_per_logical_request": self.maximum_attempts,
            "timeout_seconds": self.timeout_seconds,
            "retry_payload_byte_identical": True,
            "prompt_changed": False,
            "tool_schema_changed": False,
            "response_body_recorded": False,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.as_record())


FROZEN_TRANSPORT_RESILIENCE_POLICY = TransportResiliencePolicy()


def safe_request_metrics(payload: bytes) -> dict[str, object]:
    """Describe request structure without retaining message or tool content."""

    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P9ControllerError("cannot measure non-JSON provider request") from error
    if not isinstance(parsed, Mapping):
        raise P9ControllerError("provider request metrics require an object")
    messages = parsed.get("messages")
    tools = parsed.get("tools")
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise P9ControllerError("provider request metrics lack messages or tools")
    role_counts: dict[str, int] = {}
    message_bytes = 0
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
            raise P9ControllerError("provider request contains malformed message")
        role = str(message["role"])
        role_counts[role] = role_counts.get(role, 0) + 1
        message_bytes += len(canonical_json(message).encode("utf-8"))
    tool_bytes = sum(
        len(canonical_json(tool).encode("utf-8"))
        for tool in tools
        if isinstance(tool, Mapping)
    )
    if tool_bytes == 0 and tools:
        raise P9ControllerError("provider request contains malformed tools")
    model = parsed.get("model")
    if not isinstance(model, str) or not model:
        raise P9ControllerError("provider request metrics lack model identity")
    return {
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "request_bytes": len(payload),
        "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
        "message_count": len(messages),
        "message_role_counts": dict(sorted(role_counts.items())),
        "message_bytes": message_bytes,
        "tool_count": len(tools),
        "tool_bytes": tool_bytes,
        "request_content_recorded": False,
        "credential_recorded": False,
    }


def classify_http_status(
    status: int, *, retryable_http_statuses: frozenset[int]
) -> str:
    if status == 200:
        return "provider_http_success"
    if status in {402, 429}:
        return "provider_quota_or_rate_limit"
    if status in {401, 403}:
        return "provider_authentication_or_authorization"
    if status in retryable_http_statuses:
        return "provider_http_retryable"
    if 400 <= status < 500:
        return "provider_request_rejected"
    if 500 <= status < 600:
        return "provider_http_server_error"
    return "provider_http_unexpected"


def classify_transport_exception(error: BaseException) -> str:
    if isinstance(error, (socket.timeout, TimeoutError)):
        return "transport_exception_timeout"
    if isinstance(error, ssl.SSLError):
        return "transport_exception_tls"
    if isinstance(error, OSError):
        return "transport_exception_os"
    return "transport_exception_unsupported"


class SafeAttemptJournal:
    """Append-only-in-memory ledger atomically checkpointed after every change."""

    def __init__(
        self,
        *,
        policy: TransportResiliencePolicy,
        path: Path | None = None,
    ) -> None:
        if path is not None and path.exists():
            raise P9ControllerError("refusing to overwrite P9 transport journal")
        self.policy = policy
        self.path = path
        self.attempts: list[dict[str, object]] = []
        self._logical_request_sequence = 0
        self._last_request_digest: str | None = None

    def _write(self) -> None:
        if self.path is None:
            return
        record = self.snapshot()
        payload = (canonical_json(record) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_bytes(payload)
        temporary.chmod(0o600)
        os.replace(temporary, self.path)
        self.path.chmod(0o600)

    def append(self, row: Mapping[str, object]) -> dict[str, object]:
        request_digest = row.get("request_sha256")
        if not isinstance(request_digest, str) or len(request_digest) != 64:
            raise P9ControllerError("transport attempt lacks request digest")
        prior_retryable = bool(self.attempts) and (
            self.attempts[-1].get("status") == "transport_error"
            or self.attempts[-1].get("status")
            in self.policy.retryable_http_statuses
        )
        if request_digest != self._last_request_digest or not prior_retryable:
            self._logical_request_sequence += 1
            retry_ordinal = 0
            self._last_request_digest = request_digest
        else:
            retry_ordinal = 1 + int(self.attempts[-1]["retry_ordinal"])
        if retry_ordinal >= self.policy.maximum_attempts:
            raise P9ControllerError("transport journal retry ceiling exceeded")
        recorded = {
            "attempt_sequence": len(self.attempts) + 1,
            "logical_request_sequence": self._logical_request_sequence,
            "retry_ordinal": retry_ordinal,
            **dict(row),
        }
        unexpected = set(str(key) for key in recorded) - SAFE_ATTEMPT_FIELDS
        if unexpected:
            raise P9ControllerError("transport journal attempted to retain unknown fields")
        self.attempts.append(recorded)
        self._write()
        return recorded

    def note_retry(
        self,
        *,
        request_sha256: str,
        next_retry_ordinal: int,
        delay_seconds: int,
    ) -> None:
        if not self.attempts or self.attempts[-1].get("request_sha256") != request_sha256:
            raise P9ControllerError("transport retry annotation lacks preceding attempt")
        if not (
            self.attempts[-1].get("status") == "transport_error"
            or self.attempts[-1].get("status")
            in self.policy.retryable_http_statuses
        ):
            raise P9ControllerError("transport retry annotation followed a terminal response")
        if next_retry_ordinal != int(self.attempts[-1]["retry_ordinal"]) + 1:
            raise P9ControllerError("transport retry ordinal drifted")
        expected = self.policy.retry_delays_seconds[next_retry_ordinal - 1]
        if delay_seconds != expected:
            raise P9ControllerError("transport retry delay drifted")
        self.attempts[-1]["next_retry_ordinal"] = next_retry_ordinal
        self.attempts[-1]["next_retry_delay_seconds"] = delay_seconds
        self._write()

    def snapshot(self) -> dict[str, object]:
        return {
            "format": "dosee.p9-content-blind-transport-journal.v1",
            "policy": self.policy.as_record(),
            "policy_digest": self.policy.digest,
            "attempt_count": len(self.attempts),
            "provider_responses_received": sum(
                row.get("provider_response_received") is True for row in self.attempts
            ),
            "transport_exceptions": sum(
                row.get("provider_response_received") is False for row in self.attempts
            ),
            "request_content_recorded": False,
            "response_content_recorded": False,
            "credential_recorded": False,
            "attempts": list(self.attempts),
        }


def audit_journal_file(path: Path) -> dict[str, object]:
    """Verify journal permissions and its content-blind schema."""

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise P9ControllerError("P9 transport journal is not private")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise P9ControllerError("P9 transport journal is malformed")
    if parsed.get("format") != "dosee.p9-content-blind-transport-journal.v1":
        raise P9ControllerError("P9 transport journal format drifted")
    attempts = parsed.get("attempts")
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        raise P9ControllerError("P9 transport journal attempts are malformed")
    if parsed.get("attempt_count") != len(attempts):
        raise P9ControllerError("P9 transport journal count drifted")
    if any(
        parsed.get(field) is not False
        for field in (
            "request_content_recorded",
            "response_content_recorded",
            "credential_recorded",
        )
    ):
        raise P9ControllerError("P9 transport journal content boundary drifted")
    for row in attempts:
        if not isinstance(row, Mapping):
            raise P9ControllerError("P9 transport journal row is malformed")
        if set(str(key) for key in row) - SAFE_ATTEMPT_FIELDS:
            raise P9ControllerError("P9 transport journal contains unknown fields")
    return {
        "format": "dosee.p9-content-blind-transport-journal-audit.v1",
        "passed": True,
        "attempt_count": len(attempts),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "private_permissions": True,
        "content_fields_absent": True,
    }

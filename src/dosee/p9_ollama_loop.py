"""Audited Ollama tool loop for a future authorized P9 execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import socket
import ssl
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.error
import urllib.request

from .p9_controller import BudgetLedger, FrozenToolRegistry, P9ControllerError, ToolAction
from .p9_transport_resilience import TransportResiliencePolicy
from .p9e_contract import canonical_json, sha256_json


CHAT_URL = "https://ollama.com/api/chat"
RETRYABLE_HTTP = frozenset({502, 503, 504})


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    payload: bytes


class ChatTransport(Protocol):
    def chat(self, payload: bytes, *, timeout_seconds: int) -> HTTPResponse: ...


class OllamaCloudTransport:
    """Minimal real transport; constructing it performs no network request."""

    def __init__(self, *, api_key: str, endpoint: str = CHAT_URL) -> None:
        if not api_key.strip():
            raise P9ControllerError("Ollama API key is empty")
        if endpoint != CHAT_URL:
            raise P9ControllerError("Ollama endpoint differs from the frozen endpoint")
        self._api_key = api_key
        self.endpoint = endpoint

    def chat(self, payload: bytes, *, timeout_seconds: int) -> HTTPResponse:
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return HTTPResponse(
                    status=int(response.status),
                    headers=dict(response.headers.items()),
                    payload=response.read(),
                )
        except urllib.error.HTTPError as error:
            status = int(error.code)
            headers = dict(error.headers.items())
            error.close()
            return HTTPResponse(status=status, headers=headers, payload=b"")


def _strict_json(payload: bytes) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise P9ControllerError(f"duplicate provider JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise P9ControllerError(f"non-finite provider JSON number: {value}")

    try:
        return json.loads(payload, object_pairs_hook=unique, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P9ControllerError("provider response is not strict JSON") from error


def ollama_tools(registry: FrozenToolRegistry, family: str) -> list[dict[str, object]]:
    result = []
    for name, row in registry.tools_for(family).items():
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": row["description"],
                    "parameters": row["parameters"],
                },
            }
        )
    return result


class OllamaToolLoop:
    def __init__(
        self,
        *,
        plan: Mapping[str, object],
        family: str,
        model_id: str,
        provider_seed: int,
        tool_registry: FrozenToolRegistry,
        transport: ChatTransport,
        adapter: Any,
        retryable_http_statuses: frozenset[int] = RETRYABLE_HTTP,
        resilience_policy: TransportResiliencePolicy | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        frozen_models = {str(row["model_id"]) for row in plan["models"]}
        if model_id not in frozen_models:
            raise P9ControllerError("model is outside the frozen P9 registry")
        if provider_seed < 0:
            raise P9ControllerError("provider seed must be non-negative")
        self.plan = dict(plan)
        self.family = family
        self.model_id = model_id
        self.provider_seed = provider_seed
        self.registry = tool_registry
        self.transport = transport
        self.adapter = adapter
        if retryable_http_statuses not in {
            RETRYABLE_HTTP,
            RETRYABLE_HTTP | {500},
        }:
            raise P9ControllerError("provider-compatible HTTP retry set is unsupported")
        self.retryable_http_statuses = retryable_http_statuses
        limits = plan["turn_limits"]
        self.max_output_tokens = int(limits["single_turn_output_tokens"])
        self.retry_count = int(limits["transport_retries_per_logical_turn"])
        if frozenset(int(value) for value in limits["retryable_http_statuses"]) != RETRYABLE_HTTP:
            raise P9ControllerError("retryable Ollama statuses differ from frozen plan")
        sampling = plan["sampling"]
        if sampling.get("stream") is not False or sampling.get("think") is not False:
            raise P9ControllerError("Ollama stream/think policy differs from frozen plan")
        self.temperature = float(sampling["temperature"])
        self.top_p = float(sampling["top_p"])
        self.request_ids: list[str] = []
        self.ledger: list[dict[str, object]] = []
        if resilience_policy is not None:
            if (
                resilience_policy.retryable_http_statuses
                != self.retryable_http_statuses
                or resilience_policy.maximum_attempts != self.retry_count + 1
            ):
                raise P9ControllerError("P9 transport resilience policy is incompatible")
            self.retry_delays_seconds = resilience_policy.retry_delays_seconds
            self.resilience_policy_digest: str | None = resilience_policy.digest
        else:
            self.retry_delays_seconds = (0,) * self.retry_count
            self.resilience_policy_digest = None
        self._sleep = sleep_fn

    def request_payload(self, messages: Sequence[Mapping[str, object]]) -> bytes:
        if not messages:
            raise P9ControllerError("Ollama message history is empty")
        allowed_roles = {"system", "user", "assistant", "tool"}
        for message in messages:
            if message.get("role") not in allowed_roles:
                raise P9ControllerError("unsupported Ollama message role")
        return canonical_json(
            {
                "model": self.model_id,
                "messages": [dict(row) for row in messages],
                "tools": ollama_tools(self.registry, self.family),
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": self.max_output_tokens,
                    "seed": self.provider_seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            }
        ).encode("utf-8")

    def _one_turn(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        timeout_seconds: int,
    ) -> tuple[dict[str, object], int]:
        payload = self.request_payload(messages)
        response: HTTPResponse | None = None
        attempts: list[dict[str, object]] = []
        for attempt in range(self.retry_count + 1):
            delay_before = 0 if attempt == 0 else self.retry_delays_seconds[attempt - 1]
            if attempt > 0:
                observer = getattr(self.transport, "record_retry_delay", None)
                if callable(observer):
                    observer(
                        payload,
                        next_retry_ordinal=attempt,
                        delay_seconds=delay_before,
                    )
                if delay_before:
                    self._sleep(delay_before)
            try:
                candidate = self.transport.chat(payload, timeout_seconds=timeout_seconds)
            except (OSError, socket.timeout, ssl.SSLError) as error:
                attempts.append(
                    {
                        "attempt": attempt,
                        "delay_before_seconds": delay_before,
                        "status": "transport_error",
                        "error": type(error).__name__,
                    }
                )
                if attempt < self.retry_count:
                    continue
                raise P9ControllerError("Ollama transport retry ceiling exhausted") from error
            attempts.append(
                {
                    "attempt": attempt,
                    "delay_before_seconds": delay_before,
                    "status": candidate.status,
                    "response_sha256": hashlib.sha256(candidate.payload).hexdigest(),
                }
            )
            if (
                candidate.status in self.retryable_http_statuses
                and attempt < self.retry_count
            ):
                continue
            response = candidate
            break
        if response is None:
            raise P9ControllerError("Ollama transport produced no terminal response")
        if response.status != 200:
            raise P9ControllerError(f"Ollama HTTP status {response.status}")
        parsed = _strict_json(response.payload)
        if not isinstance(parsed, Mapping):
            raise P9ControllerError("Ollama response is not an object")
        if parsed.get("model") != self.model_id:
            raise P9ControllerError("resolved Ollama model differs from frozen identity")
        if parsed.get("done") is not True:
            raise P9ControllerError("Ollama response is not terminal")
        request_id = next(
            (
                str(value)
                for name, value in response.headers.items()
                if name.casefold() == "x-request-id" and value
            ),
            "",
        )
        if not request_id or request_id in self.request_ids:
            raise P9ControllerError("Ollama request ID is missing or duplicated")
        self.request_ids.append(request_id)
        output_tokens = parsed.get("eval_count")
        input_tokens = parsed.get("prompt_eval_count")
        if (
            not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
            or output_tokens > self.max_output_tokens
            or not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 0
        ):
            raise P9ControllerError("Ollama token accounting is invalid")
        message = parsed.get("message")
        if (
            not isinstance(message, Mapping)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content", ""), str)
        ):
            raise P9ControllerError("Ollama assistant message is malformed")
        self.ledger.append(
            {
                "request_id": request_id,
                "request_sha256": hashlib.sha256(payload).hexdigest(),
                "response_sha256": hashlib.sha256(response.payload).hexdigest(),
                "physical_attempts": attempts,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "credential_recorded": False,
                "resilience_policy_digest": self.resilience_policy_digest,
            }
        )
        return dict(message), len(attempts)

    def run(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        logical_turn_ceiling: int,
        timeout_seconds: int = 300,
    ) -> dict[str, object]:
        budget = BudgetLedger(
            logical_ceiling=logical_turn_ceiling,
            physical_ceiling=logical_turn_ceiling * (self.retry_count + 1),
        )
        history = [dict(row) for row in messages]
        while getattr(self.adapter.terminal, "decision", None) is None:
            if budget.logical_turns >= budget.logical_ceiling:
                raise P9ControllerError("logical turn ceiling reached before provider request")
            assistant, attempts = self._one_turn(history, timeout_seconds=timeout_seconds)
            budget.charge(physical_attempts=attempts)
            history.append(assistant)
            raw_calls = assistant.get("tool_calls")
            if not isinstance(raw_calls, list) or len(raw_calls) != 1:
                raise P9ControllerError("P9 requires exactly one tool call per turn")
            raw_function = raw_calls[0].get("function") if isinstance(raw_calls[0], Mapping) else None
            if not isinstance(raw_function, Mapping):
                raise P9ControllerError("Ollama tool call lacks a function")
            name = raw_function.get("name")
            arguments = raw_function.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise P9ControllerError("Ollama tool call is malformed")
            action = ToolAction(
                action_id=f"provider.{self.request_ids[-1]}.0",
                tool=name,
                arguments=dict(arguments),
            )
            result = self.adapter.execute(action)
            history.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": canonical_json(result),
                }
            )
        return {
            "format": "dosee.p9-ollama-tool-loop.v1",
            "model_id": self.model_id,
            "provider_seed": self.provider_seed,
            "terminal_decision": self.adapter.terminal.decision,
            "logical_turns": budget.logical_turns,
            "physical_attempts": budget.physical_attempts,
            "request_ids": list(self.request_ids),
            "request_ids_unique": len(self.request_ids) == len(set(self.request_ids)),
            "transcript_digest": sha256_json(history),
            "ledger": list(self.ledger),
        }

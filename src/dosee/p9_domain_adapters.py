"""Fail-closed native tool adapters for the future P9 provider loop."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from .p9_controller import (
    FrozenToolRegistry,
    IrreversibleDecisionLedger,
    P9ControllerError,
    ToolAction,
)
from .p9_sandbox import SeatbeltCommandExecutor
from .p9e_contract import sha256_json


_APPWORLD_HTTP_ERROR = re.compile(r"^Response status code is ([1-5][0-9]{2}):")


@dataclass(frozen=True)
class ToolAuditEntry:
    sequence: int
    tool: str
    action_digest: str
    result_digest: str
    terminal: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "tool": self.tool,
            "action_digest": self.action_digest,
            "result_digest": self.result_digest,
            "terminal": self.terminal,
        }


class WorkspaceToolAdapter:
    """Expose only frozen workspace tools; keep evaluator callbacks trusted."""

    def __init__(
        self,
        *,
        family: str,
        workspace_root: Path,
        tool_registry: FrozenToolRegistry,
        command_executor: SeatbeltCommandExecutor,
        evaluation_callback: Callable[[Path], Mapping[str, object]],
        read_byte_limit: int = 131072,
    ) -> None:
        if family not in {
            "iterative_ml_model_workspaces",
            "software_workspace_local_checks",
        }:
            raise P9ControllerError("workspace adapter received a non-workspace family")
        self.family = family
        self.workspace_root = workspace_root.resolve()
        self.registry = tool_registry
        self.command_executor = command_executor
        if command_executor.workspace_root != self.workspace_root:
            raise P9ControllerError("command and workspace adapter roots differ")
        if command_executor.family != family:
            raise P9ControllerError("command and workspace adapter families differ")
        self.evaluation_callback = evaluation_callback
        self.read_byte_limit = read_byte_limit
        self.terminal = IrreversibleDecisionLedger()
        self.sequence = 0
        self.audit: list[ToolAuditEntry] = []

    def _path(self, raw: object) -> Path:
        if not isinstance(raw, str):
            raise P9ControllerError("workspace path is not a string")
        resolved = (self.workspace_root / raw).resolve()
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise P9ControllerError("workspace path escapes through a symlink")
        return resolved

    def _record(self, action: ToolAction, result: Mapping[str, object]) -> None:
        self.audit.append(
            ToolAuditEntry(
                sequence=self.sequence,
                tool=action.tool,
                action_digest=action.digest,
                result_digest=sha256_json(result),
                terminal=self.terminal.decision is not None,
            )
        )

    def _evaluation(self) -> dict[str, object]:
        result = dict(self.evaluation_callback(self.workspace_root))
        expected = (
            {"metric_name", "threshold", "status", "positive"}
            if self.family == "iterative_ml_model_workspaces"
            else {"check_suite", "positive"}
        )
        if set(result) != expected or type(result.get("positive")) is not bool:
            raise P9ControllerError("trusted evaluator returned an unsafe visible schema")
        return result

    def execute(self, action: ToolAction) -> dict[str, object]:
        if self.terminal.decision is not None:
            raise P9ControllerError("tool call attempted after irreversible finalize")
        self.registry.validate(self.family, action)
        self.sequence += 1
        if action.tool == "read_file":
            path = self._path(action.arguments["path"])
            raw = path.read_bytes()
            if len(raw) > self.read_byte_limit:
                raise P9ControllerError("workspace read exceeds byte limit")
            result: dict[str, object] = {
                "path": str(path.relative_to(self.workspace_root)),
                "content": raw.decode("utf-8", errors="strict"),
            }
        elif action.tool == "apply_patch":
            path = self._path(action.arguments["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            encoding = action.arguments.get("encoding", "utf-8")
            content = str(action.arguments["content"])
            try:
                raw = (
                    content.encode("utf-8")
                    if encoding == "utf-8"
                    else base64.b64decode(content, validate=True)
                )
            except (UnicodeError, binascii.Error) as error:
                raise P9ControllerError("invalid patch content encoding") from error
            path.write_bytes(raw)
            result = {
                "message": "Patch applied successfully.",
                "path": str(path.relative_to(self.workspace_root)),
                "bytes_written": len(raw),
            }
        elif action.tool == "run_command":
            result = self.command_executor.execute(action).to_dict()
        elif action.tool in {"request_evaluation", "run_local_checks"}:
            expected_tool = (
                "request_evaluation"
                if self.family == "iterative_ml_model_workspaces"
                else "run_local_checks"
            )
            if action.tool != expected_tool:
                raise P9ControllerError("evaluator tool does not belong to task family")
            result = self._evaluation()
        elif action.tool == "finalize_task":
            decision = str(action.arguments["decision"])
            self.terminal.finalize(decision, sequence=self.sequence)
            result = {"accepted": True, "decision": decision, "irreversible": True}
        else:
            raise P9ControllerError("workspace adapter received unsupported tool")
        self._record(action, result)
        return result


class AppWorldToolAdapter:
    """Expose AppWorld public APIs without exposing Python execution or oracles."""

    def __init__(self, *, world: Any, tool_registry: FrozenToolRegistry) -> None:
        self.world = world
        self.registry = tool_registry
        allowed_apps = set(getattr(world.task, "allowed_apps", ()))
        self.allowed_apps = allowed_apps & {"spotify", "todoist"}
        if not self.allowed_apps:
            raise P9ControllerError("AppWorld task exposes no frozen public app")
        self.terminal = IrreversibleDecisionLedger()
        self.sequence = 0
        self.audit: list[ToolAuditEntry] = []

    def _record(self, action: ToolAction, result: Mapping[str, object]) -> None:
        self.audit.append(
            ToolAuditEntry(
                sequence=self.sequence,
                tool=action.tool,
                action_digest=action.digest,
                result_digest=sha256_json(result),
                terminal=self.terminal.decision is not None,
            )
        )

    def execute(self, action: ToolAction) -> dict[str, object]:
        if self.terminal.decision is not None:
            raise P9ControllerError("tool call attempted after irreversible finalize")
        self.registry.validate("appworld_playlist_workflow", action)
        self.sequence += 1
        if action.tool == "appworld_docs":
            app = str(action.arguments["app"])
            if app not in self.allowed_apps:
                raise P9ControllerError("AppWorld app is outside the native task boundary")
            docs = self.world.task.api_docs[app]
            result: dict[str, object] = {"app": app, "apis": dict(docs)}
        elif action.tool == "appworld_call":
            app = str(action.arguments["app"])
            api = str(action.arguments["api"])
            arguments = dict(action.arguments["arguments"])
            if app not in self.allowed_apps:
                raise P9ControllerError("AppWorld app is outside the native task boundary")
            docs = self.world.task.api_docs[app]
            if api not in docs:
                raise P9ControllerError("AppWorld API is not public for this task")
            try:
                response = self.world.apis[app][api](**arguments)
            except Exception as error:
                match = _APPWORLD_HTTP_ERROR.match(str(error))
                if match is None:
                    raise
                status_code = int(match.group(1))
                result = {
                    "error": "native_api_request_failed",
                    "status_code": status_code,
                    "retryable": status_code in {408, 409, 425, 429, 500, 502, 503, 504},
                }
            else:
                result = (
                    dict(response)
                    if isinstance(response, Mapping)
                    else {"result": response}
                )
        elif action.tool == "finalize_task":
            decision = str(action.arguments["decision"])
            self.terminal.finalize(decision, sequence=self.sequence)
            result = {"accepted": True, "decision": decision, "irreversible": True}
        else:
            raise P9ControllerError("AppWorld adapter received unsupported tool")
        self._record(action, result)
        return result

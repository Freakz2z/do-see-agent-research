"""Small public registry adapter for the v2 tool-schema fixture."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Mapping

from .p9_controller import P9ControllerError, ToolAction, _validate_schema


class FrozenToolRegistryV2:
    """Public, dependency-light equivalent of the frozen v2 registry wrapper."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        if payload.get("format") != "dosee.p9-localization-tool-registry.v2":
            raise P9ControllerError("unexpected v2 tool registry format")
        self.payload = dict(payload)
        policy = payload.get("path_policy")
        if not isinstance(policy, Mapping):
            raise P9ControllerError("v2 tool registry lacks path policy")
        self.path_policy = dict(policy)

    @property
    def digest(self) -> str:
        from .p9e_contract import sha256_json

        return sha256_json(self.payload)

    def tools_for(self, family: str) -> dict[str, Mapping[str, object]]:
        if family not in self.payload:
            raise P9ControllerError(f"unknown task family: {family}")
        rows = list(self.payload[family]) + list(self.payload.get("common_tools", []))
        result: dict[str, Mapping[str, object]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
                raise P9ControllerError("malformed tool registry row")
            name = str(row["name"])
            if name in result:
                raise P9ControllerError(f"duplicate tool name: {name}")
            result[name] = row
        return result

    def validate(self, family: str, action: ToolAction) -> None:
        tools = self.tools_for(family)
        if action.tool not in tools:
            raise P9ControllerError(f"tool is not frozen for family: {action.tool}")
        parameters = tools[action.tool].get("parameters")
        if not isinstance(parameters, Mapping):
            raise P9ControllerError("tool lacks parameter schema")
        _validate_schema(action.arguments, parameters, path=action.tool)
        if action.tool in {"read_file", "apply_patch"}:
            raw = action.arguments.get("path")
            if not isinstance(raw, str):
                raise P9ControllerError("workspace path must be a string")
            path = PurePosixPath(raw)
            if self.path_policy.get("reject_absolute_paths") and path.is_absolute():
                raise P9ControllerError("absolute workspace path rejected")
            if self.path_policy.get("reject_parent_traversal") and ".." in path.parts:
                raise P9ControllerError("parent traversal rejected")
        if action.tool == "run_command":
            argv = action.arguments["argv"]
            if any("\x00" in str(value) for value in argv):
                raise P9ControllerError("NUL in argv rejected")

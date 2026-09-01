"""OS-enforced command boundary for future P9 workspace tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence

from .p9_controller import FrozenToolRegistry, P9ControllerError, ToolAction


def _seatbelt_literal(path: Path) -> str:
    value = str(path.resolve())
    if "\n" in value or "\r" in value or "\x00" in value:
        raise P9ControllerError("unsafe path in sandbox policy")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class SandboxedCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    stdout_sha256: str
    stderr_sha256: str
    timed_out: bool
    sandbox_enforced: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "timed_out": self.timed_out,
            "sandbox_enforced": self.sandbox_enforced,
        }


class SeatbeltCommandExecutor:
    """Run argv-only commands while hiding home, temp, and trusted roots.

    This executor intentionally targets the frozen macOS experimental host.
    It fails closed when ``sandbox-exec`` is unavailable. The broad host roots
    are denied first; narrowly scoped workspace/dependency roots are then
    re-enabled by Seatbelt's more-specific path rules.
    """

    def __init__(
        self,
        *,
        family: str,
        workspace_root: Path,
        tool_registry: FrozenToolRegistry,
        dependency_roots: Sequence[Path] = (),
        trusted_roots: Sequence[Path] = (),
        environment: Mapping[str, str] | None = None,
        output_byte_limit: int = 65536,
    ) -> None:
        executable = shutil.which("sandbox-exec")
        if executable is None:
            raise P9ControllerError("sandbox-exec is required for P9 commands")
        self.sandbox_executable = executable
        self.family = family
        self.workspace_root = workspace_root.resolve()
        if not self.workspace_root.is_dir():
            raise P9ControllerError("sandbox workspace root does not exist")
        self.registry = tool_registry
        self.dependency_roots = tuple(path.resolve() for path in dependency_roots)
        self.trusted_roots = tuple(path.resolve() for path in trusted_roots)
        self.environment = dict(environment or {})
        allowed_environment = {
            "PYTHONPATH",
            "PYTHONDONTWRITEBYTECODE",
            "HF_HOME",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        }
        if set(self.environment) - allowed_environment or not all(
            isinstance(value, str) for value in self.environment.values()
        ):
            raise P9ControllerError("sandbox environment override is not allowlisted")
        if output_byte_limit < 1024:
            raise P9ControllerError("sandbox output limit is too small")
        self.output_byte_limit = output_byte_limit
        allowed = (self.workspace_root, *self.dependency_roots)
        for trusted in self.trusted_roots:
            if any(
                trusted == root or trusted in root.parents or root in trusted.parents
                for root in allowed
            ):
                raise P9ControllerError("trusted root overlaps an allowed sandbox root")

    def profile(self) -> str:
        home = Path.home().resolve()
        temporary = Path(os.path.realpath("/tmp"))
        clauses = [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            f"(deny file-read* (subpath {_seatbelt_literal(home)}))",
            f"(deny file-write* (subpath {_seatbelt_literal(home)}))",
            f"(deny file-read* (subpath {_seatbelt_literal(temporary)}))",
            f"(deny file-write* (subpath {_seatbelt_literal(temporary)}))",
        ]
        for root in (self.workspace_root, *self.dependency_roots):
            clauses.append(f"(allow file-read* (subpath {_seatbelt_literal(root)}))")
        clauses.append(
            f"(allow file-write* (subpath {_seatbelt_literal(self.workspace_root)}))"
        )
        for root in self.trusted_roots:
            clauses.extend(
                (
                    f"(deny file-read* (subpath {_seatbelt_literal(root)}))",
                    f"(deny file-write* (subpath {_seatbelt_literal(root)}))",
                )
            )
        return " ".join(clauses)

    def execute(self, action: ToolAction) -> SandboxedCommandResult:
        if action.tool != "run_command":
            raise P9ControllerError("sandbox executor accepts run_command only")
        self.registry.validate(self.family, action)
        argv = tuple(str(value) for value in action.arguments["argv"])
        shell_names = {"sh", "bash", "zsh", "dash", "fish", "ksh"}
        if (
            self.registry.path_policy.get("shell_string_execution") is False
            and Path(argv[0]).name in shell_names
            and len(argv) > 1
            and argv[1] in {"-c", "-lc", "-ic"}
        ):
            raise P9ControllerError("shell-string execution rejected")
        timeout = int(action.arguments.get("timeout_seconds", 20))
        command = [self.sandbox_executable, "-p", self.profile(), *argv]
        timed_out = False
        try:
            result = subprocess.run(
                command,
                cwd=self.workspace_root,
                env={**os.environ, **self.environment},
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            returncode = result.returncode
            stdout_raw = result.stdout[: self.output_byte_limit]
            stderr_raw = result.stderr[: self.output_byte_limit]
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = 124
            stdout_raw = (error.stdout or b"")[: self.output_byte_limit]
            stderr_raw = (error.stderr or b"")[: self.output_byte_limit]
        return SandboxedCommandResult(
            argv=argv,
            returncode=returncode,
            stdout=stdout_raw.decode("utf-8", errors="replace"),
            stderr=stderr_raw.decode("utf-8", errors="replace"),
            stdout_sha256=hashlib.sha256(stdout_raw).hexdigest(),
            stderr_sha256=hashlib.sha256(stderr_raw).hexdigest(),
            timed_out=timed_out,
            sandbox_enforced=True,
        )

"""Byte-audited filesystem forks for P9-E external environment bindings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import shutil
from typing import Mapping

from .p9e_contract import ForkArm, P9EContractError, canonical_json, sha256_json


@dataclass(frozen=True)
class TreeSnapshot:
    root: Path
    files: Mapping[str, str]
    digest: str


@dataclass(frozen=True)
class FilesystemForks:
    source_snapshot: TreeSnapshot
    roots: Mapping[ForkArm, Path]
    prewrite_digests: Mapping[ForkArm, str]
    postwrite_file_digests: Mapping[ForkArm, str]
    proposed_relative_path: str
    proposed_bytes_digest: str
    immediate_receipt: Mapping[str, object]
    immediate_receipt_digest: str
    trusted_root: Path
    agent_write_roots: tuple[Path, ...]


def _safe_relative_path(relative_path: str) -> PurePosixPath:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise P9EContractError("unsafe focal write path")
    return path


def snapshot_tree(root: Path) -> TreeSnapshot:
    root = root.resolve()
    if not root.is_dir():
        raise P9EContractError(f"snapshot root is not a directory: {root}")
    files: dict[str, str] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = sha256_json(files)
    return TreeSnapshot(root=root, files=files, digest=digest)


def create_filesystem_forks(
    source_root: Path,
    destination_root: Path,
    *,
    proposed_relative_path: str,
    proposed_bytes: bytes,
    immediate_receipt: Mapping[str, object],
    trusted_root: Path,
) -> FilesystemForks:
    """Copy the same pre-write state and apply the same proposed bytes to all arms.

    ``destination_root`` and its arm children must not exist.  Trusted state is
    intentionally supplied separately and may not be nested under any arm.
    """

    relative = _safe_relative_path(proposed_relative_path)
    source = source_root.resolve()
    destination = destination_root.resolve()
    trusted = trusted_root.resolve()
    if destination.exists():
        raise P9EContractError("destination root already exists")
    if not source.is_dir() or not trusted.is_dir():
        raise P9EContractError("source and trusted roots must already exist")

    source_snapshot = snapshot_tree(source)
    roots: dict[ForkArm, Path] = {}
    prewrite: dict[ForkArm, str] = {}
    postwrite: dict[ForkArm, str] = {}
    proposed_digest = hashlib.sha256(proposed_bytes).hexdigest()

    for arm in ForkArm:
        arm_root = destination / arm.value
        shutil.copytree(source, arm_root)
        roots[arm] = arm_root
        prewrite[arm] = snapshot_tree(arm_root).digest
        focal_path = arm_root.joinpath(*relative.parts)
        focal_path.parent.mkdir(parents=True, exist_ok=True)
        focal_path.write_bytes(proposed_bytes)
        postwrite[arm] = hashlib.sha256(focal_path.read_bytes()).hexdigest()

    if set(prewrite.values()) != {source_snapshot.digest}:
        raise P9EContractError("pre-write filesystem snapshots are not identical")
    if set(postwrite.values()) != {proposed_digest}:
        raise P9EContractError("proposed focal write bytes drifted across forks")

    agent_roots = tuple(path.resolve() for path in roots.values())
    for agent_root in agent_roots:
        try:
            trusted.relative_to(agent_root)
        except ValueError:
            pass
        else:
            raise P9EContractError("trusted root is nested under an agent-writable fork")

    return FilesystemForks(
        source_snapshot=source_snapshot,
        roots=roots,
        prewrite_digests=prewrite,
        postwrite_file_digests=postwrite,
        proposed_relative_path=relative.as_posix(),
        proposed_bytes_digest=proposed_digest,
        immediate_receipt=dict(immediate_receipt),
        immediate_receipt_digest=sha256_json(immediate_receipt),
        trusted_root=trusted,
        agent_write_roots=agent_roots,
    )


def audit_filesystem_forks(forks: FilesystemForks) -> dict[str, object]:
    if set(forks.roots) != set(ForkArm):
        raise P9EContractError("filesystem fork census is incomplete")
    if set(forks.prewrite_digests.values()) != {forks.source_snapshot.digest}:
        raise P9EContractError("pre-write digest drift detected")
    if set(forks.postwrite_file_digests.values()) != {forks.proposed_bytes_digest}:
        raise P9EContractError("post-write focal bytes drift detected")
    for root in forks.agent_write_roots:
        try:
            forks.trusted_root.relative_to(root)
        except ValueError:
            continue
        raise P9EContractError("trusted root became agent-writable")
    return {
        "format": "dosee.p9e-filesystem-fork-audit.v1",
        "source_tree_digest": forks.source_snapshot.digest,
        "prewrite_fork_digest": next(iter(forks.prewrite_digests.values())),
        "proposed_relative_path": forks.proposed_relative_path,
        "proposed_bytes_digest": forks.proposed_bytes_digest,
        "immediate_receipt_digest": forks.immediate_receipt_digest,
        "fork_count": len(forks.roots),
        "trusted_root_outside_agent_writes": True,
        "filesystem_fork_integrity_passed": True,
        "provider_calls": 0,
    }

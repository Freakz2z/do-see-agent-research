#!/usr/bin/env python3
"""Run the public P12 deterministic preflight without contacting Ollama."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dosee.p12_fault_conditioned_runtime import audit_fault_conditioned_runtime
from dosee.p12_fault_injection import audit_fault_injection
from dosee.p9_runtime_factory import FrozenToolRegistryV2


def main() -> None:
    registry = FrozenToolRegistryV2(
        json.loads((ROOT / "configs/p9_localization_tool_registry_v2.json").read_text())
    )
    with tempfile.TemporaryDirectory(prefix="dosee-public-p12-") as directory:
        root = Path(directory)
        fault = audit_fault_injection(
            root=root / "faults", python_executable=Path(sys.executable)
        )
        runtime = audit_fault_conditioned_runtime(
            root=root / "runtime",
            registry=registry,
            python_executable=Path(sys.executable),
        )
    result = {
        "format": "dosee.public-zero-call-preflight.v1",
        "passed": fault["passed"] and runtime["passed"],
        "fault_injection": fault,
        "runtime": runtime,
        "external_provider_calls": 0,
        "metadata_requests": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

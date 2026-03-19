#!/usr/bin/env python3
"""Helpers for launching configured local experiments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def preferred_python() -> Path:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def normalize_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_core_script(script_name: str, config: dict[str, Any]) -> None:
    command = [str(preferred_python()), str(ROOT / script_name)]
    for key, value in config.items():
        command.append(f"--{key.replace('_', '-')}")
        command.append(normalize_value(value))

    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


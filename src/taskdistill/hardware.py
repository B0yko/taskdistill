"""Hardware and machine-state facts for reports.

Hardware strings come from ``sysctl`` only; the host name and the user name are never read.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

_SYSCTL_TIMEOUT = 5.0


def _run(argv: list[str]) -> str | None:
    """Run a short command and return its stripped stdout, or None on any failure."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=_SYSCTL_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.strip()
    return text or None


def _sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def _is_macos() -> bool:
    return sys.platform == "darwin"


def hardware_info() -> dict[str, Any]:
    """Machine model, CPU and memory, e.g. ``{"model": "Mac17,4", "cpu": "Apple M5", "memory_gb": 24.0}``."""
    if not _is_macos():
        return {"model": "unknown", "cpu": platform.machine(), "memory_gb": None, "os": platform.system()}
    memsize = _sysctl("hw.memsize")
    memory_gb: float | None = None
    if memsize and memsize.isdigit():
        memory_gb = round(int(memsize) / 2**30, 1)
    os_version = _sysctl("kern.osproductversion")
    return {
        "model": _sysctl("hw.model") or "unknown",
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.machine(),
        "memory_gb": memory_gb,
        "os": f"macOS {os_version}" if os_version else "macOS",
    }


def load_average() -> list[float] | None:
    """1, 5 and 15 minute load averages, or None where the platform has none."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


_SWAP_USED = re.compile(r"used\s*=\s*([\d.]+)([KMG])", re.IGNORECASE)
_PRESSURE_FREE = re.compile(r"free percentage:\s*(\d+)%", re.IGNORECASE)
_UNIT_GB = {"K": 1 / 2**20, "M": 1 / 2**10, "G": 1.0}


def memory_pressure() -> dict[str, Any]:
    """Free-memory percentage and swap use; every field is None when it cannot be read."""
    state: dict[str, Any] = {"free_percent": None, "pressure_free_percent": None, "swap_used_gb": None}
    if not _is_macos():
        return state
    level = _sysctl("kern.memorystatus_level")
    if level and level.isdigit():
        state["free_percent"] = int(level)
    pressure = _run(["memory_pressure", "-Q"])
    if pressure:
        match = _PRESSURE_FREE.search(pressure)
        if match:
            state["pressure_free_percent"] = int(match.group(1))
    swap = _sysctl("vm.swapusage")
    if swap:
        match = _SWAP_USED.search(swap)
        if match:
            state["swap_used_gb"] = round(float(match.group(1)) * _UNIT_GB[match.group(2).upper()], 2)
    return state


def machine_state() -> dict[str, Any]:
    """Load average and memory pressure at one moment, for timing runs."""
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "load_average": load_average(),
        "memory": memory_pressure(),
    }

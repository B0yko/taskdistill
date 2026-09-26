from __future__ import annotations

import getpass
import os
import platform
import socket
import subprocess
import sys
from typing import Any

import pytest

from taskdistill import hardware


@pytest.fixture(autouse=True)
def no_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hardware facts must never come from the host name or the user name."""

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("host or user identity was read")

    monkeypatch.setattr(platform, "node", forbidden)
    monkeypatch.setattr(socket, "gethostname", forbidden)
    monkeypatch.setattr(getpass, "getuser", forbidden)
    monkeypatch.setattr(os, "getlogin", forbidden)


def fake_commands(monkeypatch: pytest.MonkeyPatch, outputs: dict[str, str | None]) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        key = argv[-1] if argv[0] == "sysctl" else " ".join(argv)
        out = outputs.get(key)
        if out is None:
            return subprocess.CompletedProcess(argv, 1, "", "unknown oid")
        return subprocess.CompletedProcess(argv, 0, out + "\n", "")

    monkeypatch.setattr(hardware.subprocess, "run", run)
    monkeypatch.setattr(hardware, "_is_macos", lambda: True)
    return calls


def test_hardware_info_from_sysctl_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_commands(
        monkeypatch,
        {
            "hw.model": "Mac17,4",
            "machdep.cpu.brand_string": "Apple M5",
            "hw.memsize": "25769803776",
            "kern.osproductversion": "26.6.2",
        },
    )
    info = hardware.hardware_info()
    assert info == {"model": "Mac17,4", "cpu": "Apple M5", "memory_gb": 24.0, "os": "macOS 26.6.2"}
    assert all(argv[:2] == ["sysctl", "-n"] for argv in calls)


def test_hardware_info_tolerates_missing_sysctl(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: Any, **__: Any) -> Any:
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(hardware.subprocess, "run", broken)
    monkeypatch.setattr(hardware, "_is_macos", lambda: True)
    info = hardware.hardware_info()
    assert info["model"] == "unknown" and info["memory_gb"] is None
    assert info["cpu"] == platform.machine()


def test_hardware_info_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_is_macos", lambda: False)
    info = hardware.hardware_info()
    assert info["model"] == "unknown"
    assert info["cpu"] == platform.machine()
    assert info["memory_gb"] is None


@pytest.mark.skipif(sys.platform != "darwin", reason="real sysctl values exist on macOS only")
def test_hardware_info_on_this_mac() -> None:
    info = hardware.hardware_info()
    assert set(info) == {"model", "cpu", "memory_gb", "os"}
    assert info["memory_gb"] and info["memory_gb"] > 0
    assert info["model"] != "unknown"


def test_load_average() -> None:
    value = hardware.load_average()
    if hasattr(os, "getloadavg"):
        assert value is not None and len(value) == 3 and all(x >= 0 for x in value)


def test_memory_pressure_parses_macos_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_commands(
        monkeypatch,
        {
            "kern.memorystatus_level": "73",
            "memory_pressure -Q": "The system has 25769803776 (1572864 pages with a page size of 16384).\n"
            "System-wide memory free percentage: 71%",
            "vm.swapusage": "total = 11264.00M  used = 10287.44M  free = 976.56M  (encrypted)",
        },
    )
    assert hardware.memory_pressure() == {"free_percent": 73, "pressure_free_percent": 71, "swap_used_gb": 10.05}


def test_memory_pressure_never_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_commands(monkeypatch, {"kern.memorystatus_level": "not a number"})
    assert hardware.memory_pressure() == {"free_percent": None, "pressure_free_percent": None, "swap_used_gb": None}

    def timeout(argv: list[str], **_: Any) -> Any:
        raise subprocess.TimeoutExpired(argv, 5)

    monkeypatch.setattr(hardware.subprocess, "run", timeout)
    assert set(hardware.memory_pressure().values()) == {None}


def test_machine_state() -> None:
    state = hardware.machine_state()
    assert set(state) == {"timestamp", "load_average", "memory"}
    assert state["timestamp"].endswith("+00:00")
    assert set(state["memory"]) == {"free_percent", "pressure_free_percent", "swap_used_gb"}

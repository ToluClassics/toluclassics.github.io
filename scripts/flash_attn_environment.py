#!/usr/bin/env python3
"""Report the environment dimensions relevant to FlashAttention-2 binaries.

The script does not download, install, or select a wheel. Its compatibility key
is a search aid that must still be checked against an official release asset.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import sysconfig
from typing import Any


def _major_minor(version: str) -> str | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    return ".".join(match.groups()) if match else None


def _python_tag() -> str:
    prefixes = {
        "cpython": "cp",
        "pypy": "pp",
        "ironpython": "ip",
        "jython": "jy",
    }
    implementation = sys.implementation.name
    prefix = prefixes.get(implementation, implementation)
    return f"{prefix}{sys.version_info.major}{sys.version_info.minor}"


def _cxx11_abi(torch_module: Any) -> tuple[bool | None, str | None]:
    public_helper = getattr(torch_module, "compiled_with_cxx11_abi", None)
    if callable(public_helper):
        return bool(public_helper()), "torch.compiled_with_cxx11_abi"

    torch_c = getattr(torch_module, "_C", None)
    private_value = getattr(torch_c, "_GLIBCXX_USE_CXX11_ABI", None)
    if private_value is not None:
        return bool(private_value), "torch._C._GLIBCXX_USE_CXX11_ABI"

    return None, None


def inspect_torch(torch_module: Any) -> dict[str, Any]:
    """Collect PyTorch build and visible-device information."""
    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    version_info = getattr(torch_module, "version", None)
    cuda_build = getattr(version_info, "cuda", None)
    rocm_build = getattr(version_info, "hip", None)
    cxx11_abi, abi_source = _cxx11_abi(torch_module)

    report: dict[str, Any] = {
        "available": True,
        "version": torch_version,
        "major_minor": _major_minor(torch_version),
        "cuda_build": cuda_build,
        "rocm_build": rocm_build,
        "cxx11_abi": cxx11_abi,
        "cxx11_abi_source": abi_source,
        "accelerator_available": False,
        "device_name": None,
        "compute_capability": None,
    }

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return report

    report["accelerator_available"] = bool(cuda.is_available())
    if not report["accelerator_available"]:
        return report

    get_device_name = getattr(cuda, "get_device_name", None)
    if callable(get_device_name):
        report["device_name"] = str(get_device_name(0))

    get_device_capability = getattr(cuda, "get_device_capability", None)
    if callable(get_device_capability):
        capability = get_device_capability(0)
        report["compute_capability"] = list(capability)

    return report


def collect_report() -> dict[str, Any]:
    """Collect interpreter, platform, and PyTorch information."""
    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "tag": _python_tag(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "wheel_platform": sysconfig.get_platform().replace("-", "_").replace(".", "_"),
        },
    }

    try:
        import torch
    except Exception as error:  # Import errors can include missing shared libraries.
        report["torch"] = {
            "available": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    else:
        report["torch"] = inspect_torch(torch)

    report["compatibility_key"] = compatibility_key(report)
    return report


def compatibility_key(report: dict[str, Any]) -> str | None:
    """Return release-search tokens, not a wheel recommendation."""
    torch_report = report.get("torch", {})
    if not torch_report.get("available"):
        return None

    tokens: list[str] = []
    cuda_build = torch_report.get("cuda_build")
    rocm_build = torch_report.get("rocm_build")
    if cuda_build and (cuda_family := _major_minor(str(cuda_build))):
        tokens.append(f"cu{cuda_family.split('.')[0]}")
    elif rocm_build and (rocm_family := _major_minor(str(rocm_build))):
        tokens.append(f"rocm{rocm_family.replace('.', '')}")

    if torch_version := torch_report.get("major_minor"):
        tokens.append(f"torch{torch_version}")

    cxx11_abi = torch_report.get("cxx11_abi")
    if cxx11_abi is not None:
        tokens.append(f"cxx11abi{str(cxx11_abi).upper()}")

    tokens.extend(
        (
            report["python"]["tag"],
            report["platform"]["wheel_platform"],
        )
    )
    return " ".join(tokens)


def format_report(report: dict[str, Any]) -> str:
    """Format a compact report for copying into an issue or debugging note."""
    python_report = report["python"]
    platform_report = report["platform"]
    torch_report = report["torch"]
    lines = [
        f"Python: {python_report['version']} ({python_report['implementation']})",
        f"Interpreter: {python_report['executable']}",
        f"Python tag: {python_report['tag']}",
        f"Platform: {platform_report['system']} {platform_report['machine']}",
        f"Wheel platform: {platform_report['wheel_platform']}",
    ]

    if not torch_report.get("available"):
        lines.append(
            "PyTorch: unavailable "
            f"({torch_report['error_type']}: {torch_report['error']})"
        )
        return "\n".join(lines)

    lines.extend(
        (
            f"PyTorch: {torch_report['version']}",
            f"PyTorch CUDA build: {torch_report['cuda_build']}",
            f"PyTorch ROCm build: {torch_report['rocm_build']}",
            f"C++11 ABI: {torch_report['cxx11_abi']}"
            + (
                f" ({torch_report['cxx11_abi_source']})"
                if torch_report["cxx11_abi_source"]
                else ""
            ),
            f"Accelerator available: {torch_report['accelerator_available']}",
        )
    )

    if torch_report["device_name"]:
        lines.append(f"Device: {torch_report['device_name']}")
    if torch_report["compute_capability"]:
        major, minor = torch_report["compute_capability"]
        lines.append(f"Compute capability: {major}.{minor}")
    if report["compatibility_key"]:
        lines.extend(
            (
                f"Release-asset search key: {report['compatibility_key']}",
                "Verify this key against an official release; it is not a wheel recommendation.",
            )
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report FlashAttention-2 binary compatibility dimensions without "
            "downloading or installing anything."
        )
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    report = collect_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

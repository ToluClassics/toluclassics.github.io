from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT_PATH = Path(__file__).with_name("flash_attn_environment.py")
SPEC = importlib.util.spec_from_file_location("flash_attn_environment", SCRIPT_PATH)
assert SPEC and SPEC.loader
flash_environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flash_environment)


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, index: int) -> str:
        self.last_device_name_index = index
        return "Example GPU"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        self.last_capability_index = index
        return (8, 9)


class FlashAttentionEnvironmentTests(unittest.TestCase):
    def test_inspects_cuda_build_and_public_abi_helper(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.6.1+cu124",
            version=SimpleNamespace(cuda="12.4", hip=None),
            cuda=FakeCuda(available=True),
            compiled_with_cxx11_abi=lambda: True,
        )

        report = flash_environment.inspect_torch(fake_torch)

        self.assertEqual(report["major_minor"], "2.6")
        self.assertEqual(report["cuda_build"], "12.4")
        self.assertTrue(report["cxx11_abi"])
        self.assertEqual(report["cxx11_abi_source"], "torch.compiled_with_cxx11_abi")
        self.assertEqual(report["device_name"], "Example GPU")
        self.assertEqual(report["compute_capability"], [8, 9])

    def test_uses_private_abi_value_only_as_a_compatibility_fallback(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.5.0",
            version=SimpleNamespace(cuda=None, hip="6.2"),
            cuda=FakeCuda(available=False),
            _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False),
        )

        report = flash_environment.inspect_torch(fake_torch)

        self.assertFalse(report["cxx11_abi"])
        self.assertEqual(
            report["cxx11_abi_source"],
            "torch._C._GLIBCXX_USE_CXX11_ABI",
        )
        self.assertFalse(report["accelerator_available"])

    def test_builds_a_search_key_without_claiming_an_asset_exists(self) -> None:
        report = {
            "python": {"tag": "cp310"},
            "platform": {"wheel_platform": "linux_x86_64"},
            "torch": {
                "available": True,
                "major_minor": "2.6",
                "cuda_build": "12.4",
                "rocm_build": None,
                "cxx11_abi": True,
            },
        }

        key = flash_environment.compatibility_key(report)

        self.assertEqual(
            key,
            "cu12 torch2.6 cxx11abiTRUE cp310 linux_x86_64",
        )


if __name__ == "__main__":
    unittest.main()

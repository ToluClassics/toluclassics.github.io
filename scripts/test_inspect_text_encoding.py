from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).with_name("inspect_text_encoding.py")
SPEC = importlib.util.spec_from_file_location("inspect_text_encoding", SCRIPT_PATH)
assert SPEC and SPEC.loader
encoding_inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(encoding_inspector)


class InspectTextEncodingTests(unittest.TestCase):
    def test_ascii_swahili_example(self) -> None:
        report = encoding_inspector.inspect_text("Habari za asubuhi")

        self.assertEqual(report["code_point_count"], 17)
        self.assertEqual(
            report["byte_lengths"],
            {"utf-8": 17, "utf-16-le": 34, "utf-32-le": 68},
        )

    def test_yoruba_example_exposes_combining_mark(self) -> None:
        report = encoding_inspector.inspect_text("Ẹ káàárọ̀")

        self.assertEqual(report["code_point_count"], 9)
        self.assertEqual(
            report["byte_lengths"],
            {"utf-8": 17, "utf-16-le": 18, "utf-32-le": 36},
        )
        self.assertEqual(report["normalization"]["NFC"]["utf-8_bytes"], 17)
        self.assertEqual(report["normalization"]["NFD"]["utf-8_bytes"], 20)
        self.assertEqual(report["code_points"][-1]["value"], "U+0300")
        self.assertEqual(report["code_points"][-1]["name"], "COMBINING GRAVE ACCENT")

    def test_amharic_example(self) -> None:
        report = encoding_inspector.inspect_text("እንደምን አደሩ")

        self.assertEqual(report["code_point_count"], 9)
        self.assertEqual(
            report["byte_lengths"],
            {"utf-8": 25, "utf-16-le": 18, "utf-32-le": 36},
        )


if __name__ == "__main__":
    unittest.main()

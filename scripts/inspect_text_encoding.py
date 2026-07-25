#!/usr/bin/env python3
"""Inspect Unicode code points, normalization, and encoded byte lengths."""

from __future__ import annotations

import argparse
import json
import unicodedata
from typing import Any


DEFAULT_EXAMPLES = (
    ("Swahili", "Habari za asubuhi"),
    ("Yorùbá", "Ẹ káàárọ̀"),
    ("Amharic", "እንደምን አደሩ"),
)


def inspect_text(text: str) -> dict[str, Any]:
    """Return encoding and normalization details for one Unicode string."""
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)
    return {
        "text": text,
        "code_point_count": len(text),
        "byte_lengths": {
            "utf-8": len(text.encode("utf-8")),
            "utf-16-le": len(text.encode("utf-16-le")),
            "utf-32-le": len(text.encode("utf-32-le")),
        },
        "normalization": {
            "NFC": {
                "text": nfc,
                "code_point_count": len(nfc),
                "utf-8_bytes": len(nfc.encode("utf-8")),
            },
            "NFD": {
                "text": nfd,
                "code_point_count": len(nfd),
                "utf-8_bytes": len(nfd.encode("utf-8")),
            },
        },
        "code_points": [
            {
                "value": f"U+{ord(character):04X}",
                "character": character,
                "name": unicodedata.name(character, "UNNAMED"),
                "utf-8_bytes": len(character.encode("utf-8")),
            }
            for character in text
        ],
    }


def format_report(label: str, report: dict[str, Any]) -> str:
    """Format a report for terminal reading."""
    lengths = report["byte_lengths"]
    nfc = report["normalization"]["NFC"]
    nfd = report["normalization"]["NFD"]
    lines = [
        f"{label}: {report['text']!r}",
        f"code points: {report['code_point_count']}",
        (
            "bytes: "
            f"UTF-8={lengths['utf-8']}, "
            f"UTF-16LE={lengths['utf-16-le']}, "
            f"UTF-32LE={lengths['utf-32-le']}"
        ),
        (
            "normalization: "
            f"NFC={nfc['code_point_count']} code points/{nfc['utf-8_bytes']} bytes, "
            f"NFD={nfd['code_point_count']} code points/{nfd['utf-8_bytes']} bytes"
        ),
        "code points:",
    ]
    for code_point in report["code_points"]:
        character = code_point["character"]
        display = "SPACE" if character == " " else character
        lines.append(
            f"  {code_point['value']} {display!r} "
            f"{code_point['name']} ({code_point['utf-8_bytes']} UTF-8 bytes)"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Unicode normalization and encoded byte lengths."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to inspect. Without arguments, inspect the article examples.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = (
        [(f"Input {index}", text) for index, text in enumerate(args.text, start=1)]
        if args.text
        else list(DEFAULT_EXAMPLES)
    )
    reports = [
        {"label": label, **inspect_text(text)}
        for label, text in examples
    ]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    print("\n\n".join(format_report(report["label"], report) for report in reports))


if __name__ == "__main__":
    main()

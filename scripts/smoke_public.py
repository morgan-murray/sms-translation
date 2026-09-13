#!/usr/bin/env python3
"""Run authenticated public API probes without storing a plaintext password."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request


CASES = (
    ("english", "russian", "The train is ten minutes late."),
    ("russian", "english", "Поезд опаздывает на десять минут."),
    ("english", "mandarin", "Please call me when you arrive."),
    ("mandarin", "english", "请在你到达时给我打电话。"),
    ("english", "korean", "The meeting starts at nine tomorrow morning."),
    ("korean", "english", "회의는 내일 아침 아홉 시에 시작합니다."),
)

MODELSETS = ("translategemma_4b_q4",)

DESTINATION_PATTERN = {
    "english": re.compile(r"[A-Za-z]"),
    "russian": re.compile(r"[\u0400-\u04ff]"),
    "mandarin": re.compile(r"[\u3400-\u9fff]"),
    "korean": re.compile(r"[\uac00-\ud7af]"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base service URL")
    parser.add_argument("--username", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get("TRANSLATION_SMOKE_PASSWORD")
    if not password:
        print("TRANSLATION_SMOKE_PASSWORD is required", file=sys.stderr)
        return 2

    credentials = base64.b64encode(f"{args.username}:{password}".encode()).decode()
    endpoint = f"{args.url.rstrip('/')}/api/v1/translate"
    failures: list[str] = []

    for modelset in MODELSETS:
        for source, dest, corpus in CASES:
            payload = json.dumps(
                {"source": source, "dest": dest, "corpus": corpus, "modelset": modelset},
                ensure_ascii=False,
            ).encode()
            request = urllib.request.Request(
                endpoint,
                data=payload,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            label = f"{modelset:24} {source:8} -> {dest:8}"
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    body = json.load(response)
                translation = body.get("translation", "")
                actual_modelset = body.get("model_used", {}).get("modelset")
                if actual_modelset != modelset:
                    raise ValueError(f"reported modelset {actual_modelset!r}")
                if not DESTINATION_PATTERN[dest].search(translation):
                    raise ValueError(f"output does not look like {dest}: {translation!r}")
                print(f"ok  {label} {body['timing_ms']:6} ms  {translation}")
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                failures.append(f"{label}: {exc}")
                print(f"FAIL {label} {exc}")

    if failures:
        print(f"\n{len(failures)} of {len(MODELSETS) * len(CASES)} checks failed")
        return 1
    print(f"\nAll {len(MODELSETS) * len(CASES)} public translation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator


def read_records(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object.")
                yield payload
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("JSON list items must be objects.")
                yield item
            return
        if isinstance(payload, dict):
            yield payload
            return
        raise ValueError("JSON input must be either an object or a list of objects.")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield dict(row)
        return

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                yield {"id": str(line_number), "text": line}
        return

    raise ValueError(f"Unsupported input extension: {path.suffix}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_records(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            return 1
        raise ValueError("JSON input must be either an object or a list of objects.")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader)

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    raise ValueError(f"Unsupported input extension: {path.suffix}")

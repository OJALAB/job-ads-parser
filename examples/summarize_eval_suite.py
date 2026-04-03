from __future__ import annotations

import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_row(label: str, metrics: dict) -> str:
    mismatch_count = len(metrics.get("mapping_mismatches", []))
    return (
        f"{label:<28} "
        f"{metrics.get('mention_f1', 0.0):>6.3f} "
        f"{metrics.get('mapping_top1_accuracy', 0.0):>6.3f} "
        f"{metrics.get('mapping_topk_recall', 0.0):>6.3f} "
        f"{metrics.get('exact_mention_match_rate', 0.0):>6.3f} "
        f"{metrics.get('exact_top1_uri_match_rate', 0.0):>6.3f} "
        f"{mismatch_count:>4}"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python summarize_eval_suite.py metrics1.json [metrics2.json ...]", file=sys.stderr)
        return 1

    print("dataset/run                  mF1   top1   topk exactM exactU mism")
    print("-" * 68)
    for raw_path in argv[1:]:
        path = Path(raw_path)
        metrics = load_metrics(path)
        print(format_row(path.stem, metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

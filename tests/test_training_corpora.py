from __future__ import annotations

import json
import unittest
from pathlib import Path

from esco_skill_batch.text_utils import normalize_text


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


class TrainingCorporaTests(unittest.TestCase):
    def test_train_corpus_has_no_description_overlap_with_eval_sets(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        train_rows = _load_jsonl(project_root / "examples" / "train_gold_skills_pl.jsonl")
        eval_rows = _load_jsonl(project_root / "examples" / "eval_gold_skills_pl.jsonl")
        hard_rows = _load_jsonl(project_root / "examples" / "eval_gold_skills_pl_hard.jsonl")

        train_descriptions = {
            normalize_text(str(row.get("description", "")))
            for row in train_rows
            if str(row.get("description", "")).strip()
        }
        eval_descriptions = {
            normalize_text(str(row.get("description", "")))
            for row in [*eval_rows, *hard_rows]
            if str(row.get("description", "")).strip()
        }

        self.assertTrue(train_descriptions)
        self.assertTrue(eval_descriptions)
        self.assertFalse(train_descriptions & eval_descriptions)


if __name__ == "__main__":
    unittest.main()

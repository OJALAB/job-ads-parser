from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from esco_skill_batch.evaluation import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_evaluate_predictions_reports_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            gold_path = base / "gold.jsonl"
            predictions_path = base / "predictions.jsonl"

            gold_rows = [
                {
                    "id": "rec-1",
                    "gold_skills": [
                        {"mention": "Python", "esco_uri": "uri:python"},
                        {"mention": "SQL", "esco_uri": "uri:sql"},
                    ],
                },
                {
                    "id": "rec-2",
                    "gold_skills": [],
                },
            ]
            prediction_rows = [
                {
                    "id": "rec-1",
                    "matches": [
                        {
                            "mention": {"text": "Python"},
                            "esco_matches": [{"concept_uri": "uri:python"}],
                        },
                        {
                            "mention": {"text": "SQL"},
                            "esco_matches": [{"concept_uri": "uri:sql"}, {"concept_uri": "uri:other"}],
                        },
                        {
                            "mention": {"text": "Excel"},
                            "esco_matches": [{"concept_uri": "uri:excel"}],
                        },
                    ],
                },
                {
                    "id": "rec-2",
                    "matches": [],
                },
            ]

            gold_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold_rows),
                encoding="utf-8",
            )
            predictions_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
                encoding="utf-8",
            )

            metrics = evaluate_predictions(gold_path, predictions_path, top_k=2)

            self.assertEqual(metrics["gold_records"], 2)
            self.assertAlmostEqual(metrics["mention_precision"], 2 / 3)
            self.assertAlmostEqual(metrics["mention_recall"], 1.0)
            self.assertAlmostEqual(metrics["mapping_top1_accuracy"], 1.0)
            self.assertAlmostEqual(metrics["mapping_topk_recall"], 1.0)
            self.assertEqual(metrics["mapping_mismatches"], [])


if __name__ == "__main__":
    unittest.main()

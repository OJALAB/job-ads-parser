from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from esco_skill_batch.gliner_training import prepare_gliner_datasets, prepare_gliner_record


class GLiNERTrainingTests(unittest.TestCase):
    def test_prepare_gliner_record_aligns_mentions_to_token_spans(self) -> None:
        record = {
            "id": "job-1",
            "description": "Szukamy osoby, ktora zna Python, zapytania SQL oraz umiejetnosci komunikacyjne.",
            "gold_skills": [
                {"mention": "Python"},
                {"mention": "zapytania SQL"},
                {"mention": "umiejetnosci komunikacyjne"},
            ],
        }

        examples, unmatched, record_id = prepare_gliner_record(
            record=record,
            text_field="description",
            skills_field="gold_skills",
            label="skill",
            keep_empty=False,
            max_tokens=None,
            window_stride=None,
        )

        self.assertEqual(record_id, "job-1")
        self.assertEqual(unmatched, [])
        self.assertEqual(len(examples), 1)
        self.assertEqual(
            examples[0]["ner"],
            [
                [5, 5, "skill"],
                [7, 8, "skill"],
                [10, 11, "skill"],
            ],
        )

    def test_prepare_gliner_record_uses_explicit_offsets(self) -> None:
        text = "Python i SQL"
        record = {
            "id": "job-2",
            "description": text,
            "gold_skills": [
                {"mention": "Python", "start": 0, "end": 6},
                {"mention": "SQL", "start": 9, "end": 12},
            ],
        }

        examples, unmatched, _ = prepare_gliner_record(
            record=record,
            text_field="description",
            skills_field="gold_skills",
            label="skill",
            keep_empty=False,
            max_tokens=None,
            window_stride=None,
        )

        self.assertEqual(unmatched, [])
        self.assertEqual(examples[0]["ner"], [[0, 0, "skill"], [2, 2, "skill"]])

    def test_prepare_gliner_record_chunks_long_records(self) -> None:
        record = {
            "id": "job-3",
            "description": "Python SQL komunikacja analiza",
            "gold_skills": [{"mention": "Python"}, {"mention": "komunikacja"}],
        }

        examples, unmatched, _ = prepare_gliner_record(
            record=record,
            text_field="description",
            skills_field="gold_skills",
            label="skill",
            keep_empty=False,
            max_tokens=2,
            window_stride=2,
        )

        self.assertEqual(unmatched, [])
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["id"], "job-3#0")
        self.assertEqual(examples[0]["ner"], [[0, 0, "skill"]])
        self.assertEqual(examples[1]["id"], "job-3#1")
        self.assertEqual(examples[1]["ner"], [[0, 0, "skill"]])

    def test_prepare_gliner_datasets_writes_train_dev_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_path = base / "gold.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "job-1",
                                "description": "Python i SQL",
                                "gold_skills": [{"mention": "Python"}, {"mention": "SQL"}],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "job-2",
                                "description": "Brak wymagan technicznych.",
                                "gold_skills": [],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "job-3",
                                "description": "kompetencje komunikacyjne",
                                "gold_skills": [{"mention": "kompetencje komunikacyjne"}],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = prepare_gliner_datasets(
                input_paths=[input_path],
                output_dir=base / "out",
                text_field="description",
                skills_field="gold_skills",
                label="skill",
                dev_ratio=0.34,
                seed=7,
                keep_empty=False,
                max_tokens=0,
                window_stride=0,
            )

            train_rows = json.loads((base / "out" / "train.json").read_text(encoding="utf-8"))
            dev_rows = json.loads((base / "out" / "dev.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["prepared_records"], 2)
            self.assertEqual(manifest["train_examples"] + manifest["dev_examples"], 2)
            self.assertEqual(len(train_rows) + len(dev_rows), 2)


if __name__ == "__main__":
    unittest.main()

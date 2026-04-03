from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from esco_skill_batch.esco import load_esco_skills, save_index
from esco_skill_batch.extractors import PassthroughExtractor
from esco_skill_batch.matching import LexicalMatcher
from esco_skill_batch.review_workflow import (
    build_finetune_corpus,
    export_review_csv,
    import_review_csv,
    prepare_review_queue,
)
from tests.helpers import write_esco_csv


def build_index(tmp_dir: str) -> Path:
    base = Path(tmp_dir)
    csv_path = base / "esco.csv"
    write_esco_csv(csv_path)
    skills = load_esco_skills(csv_path, language=None, include_knowledge=False, include_language_skills=False)
    index_dir = base / "index"
    save_index(
        output_dir=index_dir,
        skills=skills,
        source_csv=csv_path,
        language=None,
        include_knowledge=False,
        include_language_skills=False,
    )
    return index_dir


class ReviewWorkflowTests(unittest.TestCase):
    def test_prepare_review_queue_groups_mentions_across_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_path = base / "jobs.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "job-1",
                                "language": "pl",
                                "description": "Wymagane: zapytania SQL i Python.",
                                "skills_raw": ["zapytania SQL", "Python"],
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "id": "job-2",
                                "language": "pl",
                                "description": "Szukamy osoby z zapytania SQL oraz komunikacja.",
                                "skills_raw": ["zapytania SQL", "komunikacja"],
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            queue_path = base / "queue.jsonl"
            index_dir = build_index(tmp_dir)

            summary = prepare_review_queue(
                input_paths=[input_path],
                output_path=queue_path,
                extractor=PassthroughExtractor("skills_raw"),
                matcher=LexicalMatcher(index_dir),
                text_field="description",
                id_field="id",
                top_k=3,
                score_threshold=0.0,
                max_records=None,
            )

            rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
            sql_row = next(row for row in rows if row["mention_normalized"] == "zapytania sql")

            self.assertEqual(summary["queue_size"], 3)
            self.assertEqual(sql_row["occurrence_count"], 2)
            self.assertEqual(sql_row["source_record_ids"], ["job-1", "job-2"])
            self.assertTrue(sql_row["top_k_esco_candidates"])
            self.assertEqual(sql_row["top_k_esco_candidates"][0]["concept_uri"], "http://data.europa.eu/esco/skill/sql")

    def test_export_and_import_review_csv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            queue_path = base / "queue.jsonl"
            queue_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "cand-000001",
                        "mention_raw": "zapytania SQL",
                        "mention_normalized": "zapytania sql",
                        "canonical_mention": "zapytania SQL",
                        "language": "pl",
                        "occurrence_count": 2,
                        "auto_status": "needs_review",
                        "example_contexts": [{"context": "... zapytania SQL ..."}],
                        "top_k_esco_candidates": [
                            {
                                "concept_uri": "http://data.europa.eu/esco/skill/sql",
                                "preferred_label": "SQL",
                                "score": 0.7,
                                "matched_on": "label_overlap",
                            }
                        ],
                        "occurrences": [],
                        "decision": "",
                        "selected_esco_uri": "",
                        "notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            csv_path = base / "review.csv"
            reviewed_path = base / "reviewed.jsonl"

            export_review_csv(queue_path, csv_path)

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["decision"] = "accept_esco"
            rows[0]["selected_esco_uri"] = "http://data.europa.eu/esco/skill/sql"
            rows[0]["canonical_mention"] = "zapytania sql"
            rows[0]["notes"] = "accepted"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            import_review_csv(queue_path, csv_path, reviewed_path)
            reviewed_rows = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(reviewed_rows[0]["decision"], "accept_esco")
            self.assertEqual(reviewed_rows[0]["selected_esco_uri"], "http://data.europa.eu/esco/skill/sql")
            self.assertEqual(reviewed_rows[0]["canonical_mention"], "zapytania sql")
            self.assertEqual(reviewed_rows[0]["notes"], "accepted")

    def test_build_finetune_corpus_creates_aliases_and_non_overlapping_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_path = base / "jobs.jsonl"
            jobs = [
                {
                    "id": "job-1",
                    "language": "pl",
                    "description": "Wymagane: zapytania SQL i Python.",
                },
                {
                    "id": "job-2",
                    "language": "pl",
                    "description": "Szukamy osoby z programowanie w Pythonie.",
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in jobs),
                encoding="utf-8",
            )

            reviewed_queue_path = base / "reviewed_queue.jsonl"
            reviewed_rows = [
                {
                    "candidate_id": "cand-000001",
                    "mention_raw": "zapytania SQL",
                    "mention_normalized": "zapytania sql",
                    "canonical_mention": "zapytania sql",
                    "language": "pl",
                    "decision": "accept_esco",
                    "selected_esco_uri": "http://data.europa.eu/esco/skill/sql",
                    "top_k_esco_candidates": [
                        {
                            "concept_uri": "http://data.europa.eu/esco/skill/sql",
                            "preferred_label": "SQL",
                        }
                    ],
                    "occurrences": [
                        {"record_id": "job-1", "mention_text": "zapytania SQL", "start": 11, "end": 24},
                    ],
                },
                {
                    "candidate_id": "cand-000002",
                    "mention_raw": "programowanie w Pythonie",
                    "mention_normalized": "programowanie w pythonie",
                    "canonical_mention": "programowanie w Pythonie",
                    "language": "pl",
                    "decision": "accept_esco",
                    "selected_esco_uri": "http://data.europa.eu/esco/skill/python",
                    "top_k_esco_candidates": [
                        {
                            "concept_uri": "http://data.europa.eu/esco/skill/python",
                            "preferred_label": "Python",
                        }
                    ],
                    "occurrences": [
                        {"record_id": "job-2", "mention_text": "programowanie w Pythonie", "start": 16, "end": 40},
                    ],
                },
            ]
            reviewed_queue_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed_rows),
                encoding="utf-8",
            )

            output_dir = base / "corpus"
            summary = build_finetune_corpus(
                input_paths=[input_path],
                reviewed_queue_path=reviewed_queue_path,
                output_dir=output_dir,
                text_field="description",
                holdout_ratio=0.5,
                seed=7,
            )

            silver_rows = [json.loads(line) for line in (output_dir / "silver_train.jsonl").read_text(encoding="utf-8").splitlines()]
            holdout_rows = [json.loads(line) for line in (output_dir / "manual_gold_holdout.jsonl").read_text(encoding="utf-8").splitlines()]
            alias_rows = [json.loads(line) for line in (output_dir / "review_aliases.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["silver_train_records"] + summary["manual_gold_holdout_records"], 2)
            self.assertEqual(len({row["id"] for row in silver_rows} & {row["id"] for row in holdout_rows}), 0)
            self.assertEqual(len(alias_rows), 2)
            self.assertEqual(alias_rows[0]["concept_uri"].startswith("http://data.europa.eu/esco/skill/"), True)
            self.assertEqual(summary["ambiguous_occurrences"], 0)


if __name__ == "__main__":
    unittest.main()

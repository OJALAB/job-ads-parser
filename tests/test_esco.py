from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from esco_skill_batch.esco import infer_category, load_esco_skills, load_index, save_index
from tests.helpers import write_esco_csv


class EscoTests(unittest.TestCase):
    def test_infer_category_distinguishes_skill_types(self) -> None:
        self.assertEqual(infer_category("skill/competence", "sector-specific", "member-skills", ""), "skill")
        self.assertEqual(infer_category("skill/competence", "transversal", "member-skills", ""), "transversal")
        self.assertEqual(infer_category("knowledge", "sector-specific", "member-knowledge", ""), "knowledge")
        self.assertEqual(
            infer_category("knowledge", "sector-specific", "language skills and knowledge", ""),
            "language",
        )

    def test_load_esco_skills_filters_to_skills_and_transversal_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "esco.csv"
            write_esco_csv(csv_path)

            skills = load_esco_skills(
                csv_path=csv_path,
                language=None,
                include_knowledge=False,
                include_language_skills=False,
            )

            self.assertEqual([item.preferred_label for item in skills], ["Python", "SQL", "communication"])
            self.assertEqual([item.category for item in skills], ["skill", "skill", "transversal"])

    def test_load_esco_skills_can_include_knowledge_and_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "esco.csv"
            write_esco_csv(csv_path)

            skills = load_esco_skills(
                csv_path=csv_path,
                language=None,
                include_knowledge=True,
                include_language_skills=True,
            )

            self.assertEqual(
                [item.category for item in skills],
                ["skill", "skill", "transversal", "knowledge", "language"],
            )

    def test_save_and_load_index_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            csv_path = base / "esco.csv"
            write_esco_csv(csv_path)
            skills = load_esco_skills(
                csv_path,
                language=None,
                include_knowledge=False,
                include_language_skills=False,
            )

            index_dir = base / "index"
            save_index(
                output_dir=index_dir,
                skills=skills,
                source_csv=csv_path,
                language=None,
                include_knowledge=False,
                include_language_skills=False,
            )

            loaded_skills, manifest, token_index, exact_label_index = load_index(index_dir)

            self.assertEqual(len(loaded_skills), 3)
            self.assertEqual(manifest["size"], 3)
            self.assertIn("python", token_index)
            self.assertIn("communication skills", exact_label_index)

            persisted_rows = [
                json.loads(line)
                for line in (index_dir / "skills.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(persisted_rows[0]["preferred_label"], "Python")


if __name__ == "__main__":
    unittest.main()

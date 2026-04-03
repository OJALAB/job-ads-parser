from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from esco_skill_batch.esco import load_esco_skills, save_index
from esco_skill_batch.matching import LexicalMatcher, ReviewAliasMatcher
from esco_skill_batch.types import SkillMention
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


class MatchingTests(unittest.TestCase):
    def test_lexical_matcher_prefers_exact_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            matcher = LexicalMatcher(build_index(tmp_dir))

            matches = matcher.match(SkillMention(text="Python"), top_k=3, score_threshold=0.0)

            self.assertTrue(matches)
            self.assertEqual(matches[0].preferred_label, "Python")
            self.assertEqual(matches[0].matched_on, "exact_label")
            self.assertEqual(matches[0].score, 1.0)

    def test_lexical_matcher_uses_alt_label_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            matcher = LexicalMatcher(build_index(tmp_dir))

            matches = matcher.match(SkillMention(text="Python programming"), top_k=3, score_threshold=0.0)

            self.assertTrue(matches)
            self.assertEqual(matches[0].preferred_label, "Python")
            self.assertIn(matches[0].matched_on, {"exact_label", "label_overlap"})
            self.assertGreaterEqual(matches[0].score, 0.8)

    def test_lexical_matcher_respects_score_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            matcher = LexicalMatcher(build_index(tmp_dir))

            matches = matcher.match(SkillMention(text="nonexistent capability"), top_k=3, score_threshold=0.5)

            self.assertEqual(matches, [])

    def test_review_alias_matcher_prioritizes_reviewed_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_dir = build_index(tmp_dir)
            aliases_path = Path(tmp_dir) / "review_aliases.jsonl"
            aliases_path.write_text(
                json.dumps(
                    {
                        "canonical_mention": "zapytania sql",
                        "mention_normalized": "zapytania sql",
                        "concept_uri": "http://data.europa.eu/esco/skill/sql",
                        "preferred_label": "SQL",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            matcher = ReviewAliasMatcher(LexicalMatcher(index_dir), index_dir=index_dir, aliases_path=aliases_path)
            matches = matcher.match(SkillMention(text="zapytania SQL"), top_k=3, score_threshold=0.0)

            self.assertTrue(matches)
            self.assertEqual(matches[0].concept_uri, "http://data.europa.eu/esco/skill/sql")
            self.assertEqual(matches[0].matched_on, "review_alias")
            self.assertEqual(matches[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()

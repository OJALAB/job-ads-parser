from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from esco_skill_batch.esco import load_esco_skills, save_index
from esco_skill_batch.matching import LexicalMatcher
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


if __name__ == "__main__":
    unittest.main()

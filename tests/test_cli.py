from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from esco_skill_batch import cli
from esco_skill_batch.prompt_presets import BIELIK_PL_OLLAMA_PROMPT
from tests.helpers import write_esco_csv


class CliTests(unittest.TestCase):
    def test_cli_build_index_and_extract_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            csv_path = base / "esco.csv"
            write_esco_csv(csv_path)

            input_path = base / "jobs.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"id": "job-1", "description": "Need Python and communication", "skills_raw": ["Python"]}
                        ),
                        json.dumps({"id": "job-2", "description": "Need SQL", "skills_raw": ["SQL"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            index_dir = base / "index"
            output_path = base / "results.jsonl"

            build_stdout = StringIO()
            with patch("sys.stdout", build_stdout), patch(
                "sys.argv",
                [
                    "esco-skill-batch",
                    "build-index",
                    "--esco-csv",
                    str(csv_path),
                    "--output-dir",
                    str(index_dir),
                ],
            ):
                cli.main()

            build_result = json.loads(build_stdout.getvalue().strip())
            self.assertEqual(build_result["status"], "ok")
            self.assertEqual(build_result["indexed_skills"], 3)

            extract_stdout = StringIO()
            extract_stderr = StringIO()
            with patch("sys.stdout", extract_stdout), patch("sys.stderr", extract_stderr), patch(
                "sys.argv",
                [
                    "esco-skill-batch",
                    "extract-batch",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--index-dir",
                    str(index_dir),
                    "--text-field",
                    "description",
                    "--id-field",
                    "id",
                    "--extractor",
                    "passthrough",
                    "--mentions-field",
                    "skills_raw",
                    "--mapping-backend",
                    "lexical",
                    "--top-k",
                    "3",
                ],
            ):
                cli.main()

            extract_result = json.loads(extract_stdout.getvalue().strip())
            self.assertEqual(extract_result["status"], "ok")
            self.assertEqual(extract_result["processed_records"], 2)
            self.assertIn("running: job-1", extract_stderr.getvalue())
            self.assertIn("done: job-2", extract_stderr.getvalue())

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["id"], "job-1")
            self.assertEqual(rows[0]["matches"][0]["esco_matches"][0]["preferred_label"], "Python")
            self.assertEqual(rows[1]["matches"][0]["esco_matches"][0]["preferred_label"], "SQL")

    def test_make_extractor_uses_bielik_prompt_preset(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "extract-batch",
                "--input",
                "dummy.jsonl",
                "--output",
                "dummy-out.jsonl",
                "--index-dir",
                "dummy-index",
                "--extractor",
                "ollama",
                "--ollama-model",
                "bielik-pl:4.5b",
                "--ollama-prompt-preset",
                "bielik_pl",
            ]
        )

        extractor = cli._make_extractor(args)

        self.assertEqual(extractor.system_prompt, BIELIK_PL_OLLAMA_PROMPT)

    def test_make_extractor_uses_hf_token_classifier(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "extract-batch",
                "--input",
                "dummy.jsonl",
                "--output",
                "dummy-out.jsonl",
                "--index-dir",
                "dummy-index",
                "--extractor",
                "hf_token_classifier",
                "--hf-model",
                "jjzha/escoxlmr_skill_extraction",
                "--hf-aggregation-strategy",
                "simple",
                "--hf-entity-labels",
                "SKILL,TRANSVERSAL",
            ]
        )

        with patch("esco_skill_batch.cli.HFTokenClassificationExtractor") as extractor_cls:
            cli._make_extractor(args)

        extractor_cls.assert_called_once_with(
            model_name="jjzha/escoxlmr_skill_extraction",
            aggregation_strategy="simple",
            entity_labels=["SKILL", "TRANSVERSAL"],
            device=-1,
        )

    def test_cli_evaluate_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            gold_path = base / "gold.jsonl"
            prediction_path = base / "predictions.jsonl"
            gold_path.write_text(
                json.dumps(
                    {
                        "id": "rec-1",
                        "gold_skills": [{"mention": "Python", "esco_uri": "uri:python"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prediction_path.write_text(
                json.dumps(
                    {
                        "id": "rec-1",
                        "matches": [{"mention": {"text": "Python"}, "esco_matches": [{"concept_uri": "uri:python"}]}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with patch("sys.stdout", stdout), patch(
                "sys.argv",
                [
                    "esco-skill-batch",
                    "evaluate",
                    "--gold",
                    str(gold_path),
                    "--predictions",
                    str(prediction_path),
                    "--top-k",
                    "1",
                ],
            ):
                cli.main()

            result = json.loads(stdout.getvalue().strip())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["mapping_top1_accuracy"], 1.0)

    def test_cli_report_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            gold_path = base / "gold.jsonl"
            prediction_path = base / "predictions.jsonl"
            output_path = base / "report.md"
            gold_path.write_text(
                json.dumps(
                    {
                        "id": "rec-1",
                        "title": "Tester",
                        "gold_skills": [{"mention": "Python", "esco_uri": "uri:python"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prediction_path.write_text(
                json.dumps(
                    {
                        "id": "rec-1",
                        "matches": [{"mention": {"text": "Python"}, "esco_matches": [{"concept_uri": "uri:python"}]}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with patch("sys.stdout", stdout), patch(
                "sys.argv",
                [
                    "esco-skill-batch",
                    "report",
                    "--gold",
                    str(gold_path),
                    "--predictions",
                    str(prediction_path),
                    "--output",
                    str(output_path),
                ],
            ):
                cli.main()

            result = json.loads(stdout.getvalue().strip())
            self.assertEqual(result["status"], "ok")
            self.assertTrue(output_path.exists())
            self.assertIn("# Evaluation Report", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

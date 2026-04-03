from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from esco_skill_batch import cli
from esco_skill_batch.prompt_presets import BIELIK_PL_OLLAMA_PROMPT
from esco_skill_batch.runtime import ResolvedDevice
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

        with patch(
            "esco_skill_batch.cli.resolve_device_argument",
            return_value=ResolvedDevice(
                requested="auto",
                resolved="cuda:0",
                use_cpu=False,
                hf_device=0,
                cuda_index=0,
            ),
        ), patch("esco_skill_batch.cli.HFTokenClassificationExtractor") as extractor_cls:
            cli._make_extractor(args)

        extractor_cls.assert_called_once_with(
            model_name="jjzha/escoxlmr_skill_extraction",
            aggregation_strategy="simple",
            entity_labels=["SKILL", "TRANSVERSAL"],
            device="cuda:0",
        )

    def test_make_extractor_legacy_hf_device_minus_one_maps_to_cpu(self) -> None:
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
                "--hf-device",
                "-1",
            ]
        )

        with patch("esco_skill_batch.cli.HFTokenClassificationExtractor") as extractor_cls:
            cli._make_extractor(args)

        self.assertEqual(extractor_cls.call_args.kwargs["device"], "cpu")

    def test_make_extractor_uses_gliner_with_resolved_device(self) -> None:
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
                "gliner",
                "--device",
                "auto",
            ]
        )

        with patch(
            "esco_skill_batch.cli.resolve_device_argument",
            return_value=ResolvedDevice(
                requested="auto",
                resolved="cuda:0",
                use_cpu=False,
                hf_device=0,
                cuda_index=0,
            ),
        ), patch("esco_skill_batch.cli.GLiNERExtractor") as extractor_cls:
            cli._make_extractor(args)

        self.assertEqual(extractor_cls.call_args.kwargs["device"], "cuda:0")

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

    def test_cli_prepare_gliner_data_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            gold_path = base / "gold.jsonl"
            gold_path.write_text(
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
                                "description": "kompetencje komunikacyjne",
                                "gold_skills": [{"mention": "kompetencje komunikacyjne"}],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_dir = base / "gliner-data"
            stdout = StringIO()
            with patch("sys.stdout", stdout), patch(
                "sys.argv",
                [
                    "esco-skill-batch",
                    "prepare-gliner-data",
                    "--input",
                    str(gold_path),
                    "--output-dir",
                    str(output_dir),
                    "--dev-ratio",
                    "0.5",
                    "--max-tokens",
                    "0",
                    "--window-stride",
                    "0",
                ],
            ):
                cli.main()

            result = json.loads(stdout.getvalue().strip())
            self.assertEqual(result["status"], "ok")
            self.assertTrue((output_dir / "train.json").exists())
            self.assertTrue((output_dir / "dev.json").exists())

    def test_cli_train_gliner_command_delegates_to_helper(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "train-gliner",
                "--train-data",
                "train.json",
                "--dev-data",
                "dev.json",
                "--output-dir",
                "model-out",
                "--freeze-components",
                "text_encoder,labels_encoder",
            ]
        )

        with patch(
            "esco_skill_batch.cli.resolve_device_argument",
            return_value=ResolvedDevice(
                requested="auto",
                resolved="cuda:0",
                use_cpu=False,
                hf_device=0,
                cuda_index=0,
            ),
        ), patch("esco_skill_batch.cli.train_gliner_model", return_value={"status": "ok"}) as train_helper:
            cli.run_train_gliner(args)

        train_helper.assert_called_once()
        self.assertEqual(train_helper.call_args.kwargs["freeze_components"], ["text_encoder", "labels_encoder"])
        self.assertEqual(train_helper.call_args.kwargs["device"], "cuda:0")

    def test_cli_train_gliner_use_cpu_alias_maps_to_cpu(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "train-gliner",
                "--train-data",
                "train.json",
                "--output-dir",
                "model-out",
                "--use-cpu",
            ]
        )

        with patch("esco_skill_batch.cli.train_gliner_model", return_value={"status": "ok"}) as train_helper:
            cli.run_train_gliner(args)

        self.assertEqual(train_helper.call_args.kwargs["device"], "cpu")

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

from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from esco_skill_batch import cli
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


if __name__ == "__main__":
    unittest.main()

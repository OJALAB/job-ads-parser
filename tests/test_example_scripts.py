from __future__ import annotations

import unittest
from pathlib import Path


class ExampleScriptTests(unittest.TestCase):
    def test_finetune_scripts_do_not_force_use_cpu(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        for path in [
            project_root / "examples" / "run_finetune_gliner_smoke.sh",
            project_root / "examples" / "run_finetune_gliner_fair.sh",
        ]:
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("--use-cpu", content)
            self.assertIn('DEVICE="${DEVICE:-auto}"', content)
            self.assertIn('--device "$DEVICE"', content)

    def test_eval_scripts_accept_device_env(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        for path in [
            project_root / "examples" / "run_eval_gliner_suite.sh",
            project_root / "examples" / "run_eval_hf_suite.sh",
            project_root / "examples" / "run_eval_local_models_benchmark.sh",
            project_root / "examples" / "run_eval_finetuned_gliner.sh",
            project_root / "examples" / "run_prepare_review_queue.sh",
        ]:
            content = path.read_text(encoding="utf-8")
            self.assertIn("DEVICE", content)


if __name__ == "__main__":
    unittest.main()

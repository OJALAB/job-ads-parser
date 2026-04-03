from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from esco_skill_batch.runtime import resolve_device_argument, resolve_device_spec


def _make_fake_torch(*, cuda_available: bool, device_count: int = 1):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
    )
    fake_torch.device = lambda name: name
    return fake_torch


class RuntimeTests(unittest.TestCase):
    def test_resolve_device_spec_auto_prefers_cuda(self) -> None:
        fake_torch = _make_fake_torch(cuda_available=True, device_count=2)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            resolved = resolve_device_spec("auto")

        self.assertEqual(resolved.resolved, "cuda:0")
        self.assertEqual(resolved.hf_device, 0)
        self.assertFalse(resolved.use_cpu)

    def test_resolve_device_spec_auto_falls_back_to_cpu(self) -> None:
        fake_torch = _make_fake_torch(cuda_available=False)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            resolved = resolve_device_spec("auto")

        self.assertEqual(resolved.resolved, "cpu")
        self.assertEqual(resolved.hf_device, -1)
        self.assertTrue(resolved.use_cpu)

    def test_resolve_device_spec_explicit_cpu(self) -> None:
        resolved = resolve_device_spec("cpu")

        self.assertEqual(resolved.resolved, "cpu")
        self.assertTrue(resolved.use_cpu)

    def test_resolve_device_spec_raises_on_missing_cuda(self) -> None:
        fake_torch = _make_fake_torch(cuda_available=False)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
                resolve_device_spec("cuda:0")

    def test_resolve_device_argument_maps_legacy_hf_device_minus_one_to_cpu(self) -> None:
        resolved = resolve_device_argument(device="auto", hf_device_alias=-1)

        self.assertEqual(resolved.resolved, "cpu")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

import torch

import cineblur.depth_engine as depth_engine
from cineblur.depth_engine import DepthEngine


class DepthEngineTests(unittest.TestCase):
    def tearDown(self):
        release = getattr(depth_engine, "release_depth_engine", None)
        if release is not None:
            release()

    def test_shared_accessor_reuses_one_engine(self):
        get_depth_engine = getattr(depth_engine, "get_depth_engine", None)
        self.assertIsNotNone(get_depth_engine, "get_depth_engine is missing")
        first = MagicMock()
        first.device = torch.device("cuda")
        with patch.object(
            depth_engine,
            "DepthEngine",
            return_value=first,
        ) as constructor:
            self.assertIs(get_depth_engine("cuda"), first)
            self.assertIs(get_depth_engine("cuda"), first)

        constructor.assert_called_once_with(device="cuda")

    def test_cuda_oom_retries_once_at_next_lower_preset_and_logs(self):
        engine = object.__new__(DepthEngine)
        engine.device = torch.device("cuda")
        calls = []
        logs = []

        def estimate_at(frame, input_size):
            calls.append(input_size)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return torch.zeros(1, 1, 2, 2)

        engine._estimate_at = estimate_at
        with (
            patch("torch.cuda.empty_cache") as empty_cache,
            self.assertLogs("cineblur.depth_engine", level="WARNING"),
        ):
            result = engine.estimate(
                torch.zeros(1, 3, 2, 2),
                518,
                log=logs.append,
            )

        self.assertEqual(calls, [518, 392])
        self.assertEqual(tuple(result.shape), (1, 1, 2, 2))
        empty_cache.assert_called_once_with()
        self.assertTrue(any("518" in line and "392" in line for line in logs))

    def test_cuda_oom_fast_preset_falls_back_to_turbo(self):
        engine = object.__new__(DepthEngine)
        engine.device = torch.device("cuda")
        calls = []

        def estimate_at(frame, input_size):
            calls.append(input_size)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return torch.zeros(1, 1, 2, 2)

        engine._estimate_at = estimate_at
        with (
            patch("torch.cuda.empty_cache"),
            self.assertLogs("cineblur.depth_engine", level="WARNING"),
        ):
            engine.estimate(torch.zeros(1, 3, 2, 2), 392)

        self.assertEqual(calls, [392, 336])

    def test_depth_engine_returns_finite_full_resolution_depth(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the depth smoke test")
        engine = DepthEngine(device="cuda")
        self.addCleanup(engine.release)
        self.assertEqual(next(engine._model.parameters()).dtype, torch.float16)
        frame = torch.rand(1, 3, 128, 160, device=engine.device)

        depth = engine.estimate(frame, input_size=336)

        self.assertEqual(tuple(depth.shape), (1, 1, 128, 160))
        self.assertTrue(torch.isfinite(depth).all())


if __name__ == "__main__":
    unittest.main()

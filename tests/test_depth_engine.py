import unittest

import torch

from cineblur.depth_engine import DepthEngine


class DepthEngineTests(unittest.TestCase):
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

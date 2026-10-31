import unittest

import torch
from cineblur.depth_ops import depth_range, normalize_depth, smooth_range


class DepthOpsTests(unittest.TestCase):
    def test_depth_range_uses_percentiles_and_normalization(self):
        depth = torch.arange(1.0, 101.0).reshape(1, 10, 10)
        low, high = depth_range(depth, sample_step=1)
        self.assertAlmostEqual(low, 1.99, places=5)
        self.assertAlmostEqual(high, 99.01, places=5)
        out = normalize_depth(depth, low, high)
        self.assertEqual(out.shape, depth.shape)
        self.assertTrue(torch.isclose(out.min(), torch.tensor(0.0)))
        self.assertTrue(torch.isclose(out.max(), torch.tensor(1.0)))

    def test_invert_depth_swaps_black_and_white(self):
        depth = torch.tensor([[[0.0, 1.0]]])
        normal = normalize_depth(depth, 0.0, 1.0)
        inverted = normalize_depth(depth, 0.0, 1.0, invert=True)
        self.assertTrue(torch.equal(normal, 1.0 - inverted))

    def test_smooth_range_uses_previous_range(self):
        self.assertEqual(
            smooth_range((0.0, 10.0), (10.0, 20.0), alpha=0.85),
            (1.5, 11.5),
        )

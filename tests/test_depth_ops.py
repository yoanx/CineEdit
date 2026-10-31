import unittest

import torch
from cineblur.depth_ops import (
    depth_range,
    is_processed_depth_source,
    normalize_depth,
    temporal_blend,
)


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

    def test_temporal_blend_only_smooths_static_pixels(self):
        current = torch.tensor([[[[1.0, 0.5]]]])
        previous = torch.tensor([[[[0.0, 0.5]]]])
        static = torch.tensor([[[[True, False]]]])

        out = temporal_blend(current, previous, static, alpha=0.75)

        self.assertTrue(torch.isclose(out[0, 0, 0, 0], torch.tensor(0.25)))
        self.assertTrue(torch.isclose(out[0, 0, 0, 1], torch.tensor(0.5)))

    def test_detects_already_blurred_depth_source(self):
        self.assertTrue(
            is_processed_depth_source(
                r"C:\Videos\Valorant 2026.10.20.DVR_blur.mp4"
            )
        )
        self.assertFalse(
            is_processed_depth_source(
                r"C:\Videos\Valorant 2026.10.20.DVR.mp4"
            )
        )

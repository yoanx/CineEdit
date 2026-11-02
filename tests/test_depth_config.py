from __future__ import annotations

import unittest

from cinedit.config import DepthConfig


class DepthConfigTests(unittest.TestCase):
    def test_depth_input_sizes_are_model_multiples(self):
        self.assertEqual(DepthConfig(quality="turbo").input_size(), 336)
        self.assertEqual(DepthConfig(quality="fast").input_size(), 392)
        self.assertEqual(DepthConfig(quality="fine").input_size(), 518)
        self.assertTrue(
            all(
                DepthConfig(quality=q).input_size() % 14 == 0
                for q in ("turbo", "fast", "fine")
            )
        )

    def test_depth_output_name_uses_depth_suffix(self):
        self.assertTrue(
            DepthConfig().output_name("clip.mp4").endswith("clip_depth.mp4")
        )

    def test_depth_validate_accepts_supported_options(self):
        for quality in ("turbo", "fast", "fine"):
            for codec in ("auto", "h264_nvenc", "libx264"):
                with self.subTest(quality=quality, codec=codec):
                    DepthConfig(quality=quality, codec=codec).validate()

    def test_depth_validate_rejects_unsupported_quality(self):
        with self.assertRaisesRegex(
            ValueError, "quality must be turbo, fast, or fine"
        ):
            DepthConfig(quality="draft").validate()

    def test_depth_validate_rejects_unsupported_codec(self):
        with self.assertRaisesRegex(ValueError, "unsupported depth codec"):
            DepthConfig(codec="vp9").validate()

    def test_depth_validate_rejects_invalid_range_alpha(self):
        with self.assertRaisesRegex(ValueError, "range_alpha"):
            DepthConfig(range_alpha=1.1).validate()

if __name__ == "__main__":
    unittest.main()

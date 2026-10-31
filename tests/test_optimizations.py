from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import cineblur.preview as preview
from cineblur.config import BlurConfig
from cineblur.media import Decoder
from cineblur.pipeline import MotionBlurRenderer
from cineblur.webui import Handler, QuietThreadingHTTPServer, build_cfg


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "out", "demo_src.mp4")


class QualityPresetTests(unittest.TestCase):
    def test_standard_and_fine_presets_cap_flow_iterations(self):
        default = BlurConfig()
        standard = build_cfg({"quality": "std"})
        fine = build_cfg({"quality": "fine"})

        self.assertEqual(default.flow_iters, 8)
        self.assertEqual(standard.flow_iters, 8)
        self.assertEqual(fine.flow_iters, 12)


class RenderFastPathTests(unittest.TestCase):
    def test_decoder_closes_both_ffmpeg_pipes(self):
        decoder = Decoder(DEMO, 64, 64, dur=0.1)
        frame = decoder.read()
        decoder.close()

        self.assertTrue(frame.flags.writeable)
        self.assertTrue(decoder.proc.stdout.closed)
        self.assertTrue(decoder.proc.stderr.closed)

    def test_zero_shutter_angle_skips_flow_engine(self):
        cfg = BlurConfig(
            angle=0,
            samples=16,
            device="cpu",
            codec="libx264",
            trim_end=0.2,
        )

        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "zero-angle.mp4")
            with patch(
                "cineblur.pipeline.FlowEngine",
                side_effect=AssertionError("flow engine should not be constructed"),
            ):
                stats = MotionBlurRenderer(cfg).render(DEMO, dst)

            self.assertGreater(stats["frames"], 0)
            self.assertTrue(os.path.isfile(dst))


class PreviewTests(unittest.TestCase):
    def test_original_preview_decodes_one_frame(self):
        frame = np.full((64, 64, 3), 127, dtype=np.uint8)
        preview._STATE.clear()

        with patch(
            "cineblur.preview._extract_frames",
            return_value=frame[None].copy(),
        ) as extract:
            result = preview.render_original(DEMO, 0.45, max_height=64)

        self.assertEqual(extract.call_args.args[4], 1)
        self.assertEqual(result.shape, frame.shape)


class EventStreamTests(unittest.TestCase):
    def test_aborted_browser_connection_is_ignored(self):
        handler = object.__new__(Handler)
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler.wfile = type(
            "BrokenWriter",
            (),
            {"write": lambda self, data: (_ for _ in ()).throw(
                ConnectionAbortedError("browser closed"))},
        )()

        handler._events(0)

    def test_server_ignores_aborted_connection_at_request_boundary(self):
        server = QuietThreadingHTTPServer(("127.0.0.1", 0), Handler)
        try:
            with patch("sys.stderr"):
                try:
                    raise ConnectionAbortedError("browser closed")
                except ConnectionAbortedError:
                    server.handle_error(None, ("127.0.0.1", 0))
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

import cinedit.preview as preview
from cinedit.config import BlurConfig
from cinedit.media import Decoder, unique_output_path
from cinedit.pipeline import MotionBlurRenderer
from cinedit.webui import Handler, QuietThreadingHTTPServer, build_cfg


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "out", "demo_src.mp4")


class BlockingStderrStdout:
    def __init__(self, process, frame):
        self.process = process
        self.frame = frame

    def read(self, size=-1):
        if not self.process.stderr_drained.wait(timeout=0.5):
            return b""
        return self.frame[:size]

    def close(self):
        pass


class BlockingStderrStderr:
    def __init__(self, process):
        self.process = process
        self.sent = False

    def read(self, size=-1):
        if self.sent:
            return b""
        self.sent = True
        self.process.stderr_drained.set()
        return b"decoder warning\n"

    def close(self):
        pass


class BlockingStderrProcess:
    def __init__(self, frame):
        self.stderr_drained = threading.Event()
        self.stdout = BlockingStderrStdout(self, frame)
        self.stderr = BlockingStderrStderr(self)
        self.returncode = 0

    def close(self):
        pass

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


class QualityPresetTests(unittest.TestCase):
    def test_standard_and_fine_presets_cap_flow_iterations(self):
        default = BlurConfig()
        standard = build_cfg({"quality": "std"})
        fine = build_cfg({"quality": "fine"})

        self.assertEqual(default.flow_iters, 8)
        self.assertEqual(standard.flow_iters, 8)
        self.assertEqual(fine.flow_iters, 12)

    def test_build_cfg_uses_selected_compute_device(self):
        cfg = build_cfg({"quality": "fast", "device": "cpu"})

        self.assertEqual(cfg.device, "cpu")


class OutputPathTests(unittest.TestCase):
    def test_existing_render_output_gets_next_numbered_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "clip_blur.mp4")
            with open(first, "wb"):
                pass

            second = unique_output_path(first)
            self.assertEqual(second, os.path.join(tmp, "clip_blur_2.mp4"))

            with open(second, "wb"):
                pass
            third = unique_output_path(first)

        self.assertEqual(third, os.path.join(tmp, "clip_blur_3.mp4"))


class RenderFastPathTests(unittest.TestCase):
    def test_decoder_drains_stderr_so_first_frame_is_not_blocked(self):
        frame = np.full((2, 2, 3), 127, dtype=np.uint8)
        proc = BlockingStderrProcess(frame.tobytes())

        with (
            patch("cinedit.media.find_exe", return_value="ffmpeg"),
            patch("cinedit.media.subprocess.Popen", return_value=proc),
        ):
            decoder = Decoder("corrupt.mp4", 2, 2)
            try:
                result = decoder.read()
            finally:
                decoder.close()

        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, frame)

    def test_motion_blur_cancel_skips_engine_initialization(self):
        renderer = MotionBlurRenderer(BlurConfig(device="cpu"))

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(renderer, "_ensure_engine") as ensure_engine:
                stats = renderer.render(
                    DEMO,
                    os.path.join(tmp, "cancelled.mp4"),
                    cancel=lambda: True,
                )

        self.assertTrue(stats["cancelled"])
        self.assertEqual(stats["frames"], 0)
        ensure_engine.assert_not_called()

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
                "cinedit.pipeline.FlowEngine",
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
            "cinedit.preview._extract_frames",
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

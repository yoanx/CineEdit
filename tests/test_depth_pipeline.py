from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from queue import Queue as RealQueue
from unittest.mock import patch

import torch

import cineblur.depth_pipeline as depth_pipeline
from cineblur.config import DepthConfig
from cineblur.depth_pipeline import DepthRenderer
from cineblur.media import Decoder, find_exe, probe


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "out", "demo_src.mp4")


class ConstantEngine:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []
        self.released = False

    def estimate(self, frame, input_size):
        self.calls.append(input_size)
        pixels = frame.shape[-2] * frame.shape[-1]
        return torch.linspace(
            0.0,
            1.0,
            pixels,
            device=frame.device,
        ).reshape(1, 1, frame.shape[-2], frame.shape[-1])

    def release(self):
        self.released = True


class FailingEncoder:
    instances = []

    def __init__(self, *args, **kwargs):
        self.frames_written = 0
        self.closed = False
        self.__class__.instances.append(self)

    def write(self, frame):
        self.frames_written += 1
        raise OSError("simulated writer failure")

    def close(self):
        self.closed = True
        return True, ""


class RecordingEncoder:
    instances = []

    def __init__(self, *args, **kwargs):
        self.frames_written = 0
        self.closed = False
        self.__class__.instances.append(self)

    def write(self, frame):
        self.frames_written += 1

    def close(self):
        self.closed = True
        return True, ""


class ObservedQueue(RealQueue):
    def __init__(self, maxsize, stop_events):
        super().__init__(maxsize)
        self.stop_events = stop_events
        self.sentinel_stop_states = []

    def put(self, item, block=True, timeout=None):
        if item is None:
            self.sentinel_stop_states.append(self.stop_events[-1].is_set())
        return super().put(item, block=block, timeout=timeout)


class DepthPipelineTests(unittest.TestCase):
    def test_renderer_preserves_video_timing_and_writes_grayscale_without_audio(self):
        ffmpeg = find_exe("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is required")

        cfg = DepthConfig(
            quality="fast",
            output_height=64,
            device="cpu",
            codec="libx264",
            crf=0,
            trim_end=0.2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source-with-audio.mp4")
            dst = os.path.join(tmp, "depth.mp4")
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    DEMO,
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=1",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    source,
                ],
                check=True,
            )
            source_info = probe(source)
            self.assertTrue(source_info.has_audio)

            engine = ConstantEngine()
            progress = []
            logs = []
            renderer = DepthRenderer(cfg, engine=engine)
            try:
                stats = renderer.render(
                    source,
                    dst,
                    progress=progress.append,
                    log=logs.append,
                )
            finally:
                renderer.close()

            self.assertTrue(engine.released)
            self.assertEqual(engine.calls, [392] * 6)
            self.assertEqual(
                set(stats),
                {"frames", "seconds", "fps", "output", "size", "profile"},
            )
            self.assertEqual(stats["frames"], 6)
            self.assertEqual(stats["output"], dst)
            self.assertTrue(dst.lower().endswith(".mp4"))
            self.assertTrue(os.path.isfile(dst))
            self.assertEqual(progress[-1]["done"], stats["frames"])
            self.assertEqual(progress[-1]["total"], 6)
            self.assertIn("Completed", logs[-1])

            info = probe(dst)
            self.assertFalse(info.has_audio)
            self.assertEqual(info.fps, source_info.fps)
            self.assertEqual(info.nb_frames, 6)
            self.assertAlmostEqual(info.duration, 0.2, delta=0.03)
            self.assertEqual(stats["size"], f"{info.width}x{info.height}")
            self.assertEqual(info.height, 64)
            self.assertEqual(info.codec, "h264")
            self.assertEqual(info.pix_fmt, "yuv420p")
            self.assertIn("dec_ms", stats["profile"])
            self.assertIn("enc_ms", stats["profile"])
            self.assertEqual(stats["profile"]["frames_profiled"], stats["frames"])

            decoder = Decoder(dst, info.width, info.height)
            frames = []
            try:
                while True:
                    frame = decoder.read()
                    if frame is None:
                        break
                    frames.append(frame)
            finally:
                decoder.close()

            self.assertEqual(len(frames), 6)
            channel_spread = max(
                int((frame.max(axis=2) - frame.min(axis=2)).max())
                for frame in frames
            )
            self.assertLessEqual(channel_spread, 2)
            self.assertGreater(int(frames[0][..., 0].max() - frames[0][..., 0].min()), 20)

    def test_writer_failure_stops_after_first_failed_write(self):
        if not find_exe("ffmpeg"):
            self.skipTest("ffmpeg is required")

        cfg = DepthConfig(
            quality="fast",
            output_height=64,
            device="cpu",
            codec="libx264",
            trim_end=0.5,
        )
        FailingEncoder.instances.clear()
        engine = ConstantEngine()
        renderer = DepthRenderer(cfg, engine=engine)

        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "failed-depth.mp4")
            try:
                with patch("cineblur.depth_pipeline.Encoder", FailingEncoder):
                    with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                        renderer.render(DEMO, dst)
            finally:
                renderer.close()

        self.assertEqual(len(FailingEncoder.instances), 1)
        encoder = FailingEncoder.instances[0]
        self.assertEqual(encoder.frames_written, 1)
        self.assertTrue(encoder.closed)
        self.assertLess(len(engine.calls), 15)

    def test_normal_completion_queues_writer_sentinel_before_stop(self):
        if not find_exe("ffmpeg"):
            self.skipTest("ffmpeg is required")

        cfg = DepthConfig(
            quality="fast",
            output_height=64,
            device="cpu",
            codec="libx264",
            trim_end=0.2,
        )
        stop_events = []
        queues = []
        real_put = depth_pipeline._put

        def tracked_put(queue, item, stop):
            stop_events.append(stop)
            return real_put(queue, item, stop)

        def make_queue(maxsize=0):
            if len(queues) == 1:
                observed = ObservedQueue(maxsize, stop_events)
                queues.append(observed)
                return observed
            queue = RealQueue(maxsize)
            queues.append(queue)
            return queue

        RecordingEncoder.instances.clear()
        engine = ConstantEngine()
        renderer = DepthRenderer(cfg, engine=engine)
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "normal-depth.mp4")
            try:
                with (
                    patch("cineblur.depth_pipeline._put", side_effect=tracked_put),
                    patch("cineblur.depth_pipeline.Queue", side_effect=make_queue),
                    patch("cineblur.depth_pipeline.Encoder", RecordingEncoder),
                ):
                    stats = renderer.render(DEMO, dst)
            finally:
                renderer.close()

        encoder = RecordingEncoder.instances[0]
        self.assertEqual(stats["frames"], 6)
        self.assertEqual(encoder.frames_written, 6)
        self.assertTrue(encoder.closed)
        self.assertEqual(queues[1].sentinel_stop_states, [False])


if __name__ == "__main__":
    unittest.main()

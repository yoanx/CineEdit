from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import cineblur.preview as preview
import cineblur.webui as webui
from cineblur.config import DepthConfig
from cineblur.depth_engine import release_depth_engine


class FakeDepthEngine:
    instances = []

    def __init__(self, device="auto"):
        self.device = torch.device("cpu")
        self.calls = []
        self.__class__.instances.append(self)

    def estimate(self, frame, input_size):
        self.calls.append(input_size)
        height, width = frame.shape[-2:]
        return torch.linspace(
            0.0,
            1.0,
            height * width,
            device=frame.device,
        ).reshape(1, 1, height, width)


class ImmediateThread:
    def __init__(self, target, daemon=None):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class TrackingLock:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True


class DepthWebUITests(unittest.TestCase):
    def tearDown(self):
        release_depth_engine()
        preview._DEPTH_FRAME_CACHE.clear()
        preview._DEPTH_OUTPUT_CACHE.clear()
        webui.JOB.update(
            running=False,
            log=[],
            result=None,
            error=None,
            output="",
            progress={"done": 0, "total": 0, "fps": 0.0, "eta": 0.0},
        )

    def test_build_depth_cfg_maps_speed_preset(self):
        build_depth_cfg = getattr(webui, "build_depth_cfg", None)
        self.assertIsNotNone(build_depth_cfg, "build_depth_cfg is missing")

        cfg = build_depth_cfg(
            {
                "mode": "depth",
                "quality": "turbo",
                "invert": "1",
                "preview": "720",
            }
        )

        self.assertEqual(cfg.quality, "turbo")
        self.assertEqual(cfg.input_size(), 336)
        self.assertTrue(cfg.invert)
        self.assertEqual(cfg.output_height, 720)

    def test_default_output_preserves_blur_and_uses_depth_suffix(self):
        default_output = getattr(webui, "default_output", None)
        self.assertIsNotNone(default_output, "default_output is missing")

        self.assertTrue(
            default_output("clip.mp4", "depth").endswith("clip_depth.mp4")
        )
        self.assertTrue(
            default_output("clip.mp4", "blur").endswith("clip_blur.mp4")
        )

    def test_depth_preview_caches_frame_and_reuses_it_for_inversion(self):
        render_depth_preview = getattr(preview, "render_depth_preview", None)
        self.assertIsNotNone(
            render_depth_preview,
            "render_depth_preview is missing",
        )
        preview._DEPTH_FRAME_CACHE.clear()
        preview._DEPTH_OUTPUT_CACHE.clear()
        FakeDepthEngine.instances.clear()
        engine = FakeDepthEngine()
        frame = np.array(
            [[[[0, 64, 128], [255, 192, 128]], [[10, 20, 30], [40, 50, 60]]]],
            dtype=np.uint8,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            with (
                patch(
                    "cineblur.preview.get_depth_engine",
                    return_value=engine,
                ) as accessor,
                patch("cineblur.preview._size_of", return_value=(2, 2)),
                patch("cineblur.preview._fps_of", return_value=10.0),
                patch(
                    "cineblur.preview._extract_frames",
                    return_value=frame,
                ) as extract,
            ):
                normal = render_depth_preview(
                    source.name,
                    0.2,
                    DepthConfig(quality="turbo", device="cpu"),
                    max_height=2,
                )
                cached = render_depth_preview(
                    source.name,
                    0.2,
                    DepthConfig(quality="turbo", device="cpu"),
                    max_height=2,
                )
                inverted = render_depth_preview(
                    source.name,
                    0.2,
                    DepthConfig(quality="turbo", invert=True, device="cpu"),
                    max_height=2,
                )

        self.assertEqual(extract.call_count, 1)
        self.assertEqual(len(FakeDepthEngine.instances), 1)
        self.assertEqual(accessor.call_count, 2)
        accessor.assert_called_with("cpu")
        self.assertEqual(FakeDepthEngine.instances[0].calls, [336, 336])
        self.assertEqual(normal.shape, (2, 2, 3))
        np.testing.assert_array_equal(normal, cached)
        self.assertFalse(np.array_equal(normal, inverted))

    def test_depth_job_routes_to_depth_renderer_and_output(self):
        webui.JOB.update(
            running=False,
            log=[],
            result=None,
            error=None,
            output="",
            progress={"done": 0, "total": 0, "fps": 0.0, "eta": 0.0},
        )
        renderer = MagicMock()
        renderer.render.return_value = {"frames": 1, "output": "ignored"}

        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            expected_output = os.path.splitext(source.name)[0] + "_depth.mp4"
            with (
                patch.object(webui.threading, "Thread", ImmediateThread),
                patch.object(
                    webui,
                    "DepthRenderer",
                    create=True,
                    return_value=renderer,
                ) as depth_renderer,
                patch.object(webui, "MotionBlurRenderer") as blur_renderer,
            ):
                result = webui.start_job(
                    {
                        "mode": "depth",
                        "src": source.name,
                        "quality": "turbo",
                        "invert": True,
                        "preview": 720,
                        "codec": "libx264",
                    }
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], expected_output)
        depth_renderer.assert_called_once()
        blur_renderer.assert_not_called()
        cfg = depth_renderer.call_args.args[0]
        self.assertEqual(cfg.quality, "turbo")
        self.assertTrue(cfg.invert)
        self.assertEqual(cfg.output_height, 720)
        self.assertEqual(cfg.codec, "libx264")
        self.assertEqual(
            renderer.render.call_args.args,
            (source.name, expected_output),
        )
        self.assertEqual(
            set(renderer.render.call_args.kwargs),
            {"progress", "cancel", "log"},
        )

    def test_start_job_preserves_existing_output_file(self):
        webui.JOB.update(
            running=False,
            log=[],
            result=None,
            error=None,
            output="",
            progress={"done": 0, "total": 0, "fps": 0.0, "eta": 0.0},
        )
        renderer = MagicMock()
        renderer.render.return_value = {"frames": 1, "output": "ignored"}

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "clip.mp4")
            existing = os.path.join(tmp, "clip_blur.mp4")
            expected = os.path.join(tmp, "clip_blur_2.mp4")
            for path in (source, existing):
                with open(path, "wb"):
                    pass

            with (
                patch.object(webui.threading, "Thread", ImmediateThread),
                patch.object(
                    webui,
                    "MotionBlurRenderer",
                    return_value=renderer,
                ),
            ):
                result = webui.start_job(
                    {
                        "mode": "blur",
                        "src": source,
                        "dst": existing,
                        "quality": "turbo",
                    }
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], expected)
        self.assertEqual(renderer.render.call_args.args, (source, expected))

    def test_start_job_can_overwrite_when_preserve_is_disabled(self):
        webui.JOB.update(
            running=False,
            log=[],
            result=None,
            error=None,
            output="",
            progress={"done": 0, "total": 0, "fps": 0.0, "eta": 0.0},
        )
        renderer = MagicMock()
        renderer.render.return_value = {"frames": 1, "output": "ignored"}

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "clip.mp4")
            existing = os.path.join(tmp, "clip_blur.mp4")
            for path in (source, existing):
                with open(path, "wb"):
                    pass

            with (
                patch.object(webui.threading, "Thread", ImmediateThread),
                patch.object(
                    webui,
                    "MotionBlurRenderer",
                    return_value=renderer,
                ),
            ):
                result = webui.start_job(
                    {
                        "mode": "blur",
                        "src": source,
                        "dst": existing,
                        "quality": "turbo",
                        "preserve_outputs": False,
                    }
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], existing)
        self.assertEqual(renderer.render.call_args.args, (source, existing))

    def test_depth_job_closes_renderer_when_render_raises(self):
        webui.JOB.update(
            running=False,
            log=[],
            result=None,
            error=None,
            output="",
            progress={"done": 0, "total": 0, "fps": 0.0, "eta": 0.0},
        )
        renderer = MagicMock()
        renderer.render.side_effect = RuntimeError("render exploded")

        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            with (
                patch.object(webui.threading, "Thread", ImmediateThread),
                patch.object(
                    webui,
                    "DepthRenderer",
                    return_value=renderer,
                ),
            ):
                result = webui.start_job(
                    {
                        "mode": "depth",
                        "src": source.name,
                        "quality": "fast",
                    }
                )

        self.assertTrue(result["ok"])
        self.assertEqual(webui.JOB["error"], "render exploded")
        renderer.close.assert_called_once_with()
        self.assertFalse(webui.JOB["running"])

    def test_env_reports_depth_model_availability(self):
        with patch.object(
            webui,
            "depth_model_available",
            return_value=True,
        ):
            self.assertTrue(webui.env_info()["depth_model"])

        with patch.object(
            webui,
            "depth_model_available",
            return_value=False,
        ):
            self.assertFalse(webui.env_info()["depth_model"])

    def test_depth_frame_routes_through_depth_preview_under_preview_lock(self):
        webui.JOB["running"] = False
        cfg = DepthConfig(quality="turbo", invert=True, output_height=540)
        lock = TrackingLock()
        handler = object.__new__(webui.Handler)
        handler._send = MagicMock()
        query = {
            "src": ["clip.mp4"],
            "t": ["1.25"],
            "h": ["360"],
            "mode": ["depth"],
            "quality": ["turbo"],
            "invert": ["1"],
            "preview": ["540"],
        }
        image = MagicMock()
        image.save.side_effect = (
            lambda buf, fmt, quality: buf.write(b"jpeg")
        )

        with (
            patch.object(webui.os.path, "exists", return_value=True),
            patch.object(webui, "_PREVIEW_LOCK", lock),
            patch.object(webui, "build_depth_cfg", return_value=cfg) as build_depth,
            patch.object(webui, "build_cfg") as build_blur,
            patch(
                "cineblur.preview.render_depth_preview",
                return_value=np.zeros((2, 2, 3), dtype=np.uint8),
            ) as render_depth,
            patch(
                "cineblur.preview.render_preview",
            ) as render_blur,
            patch("PIL.Image.fromarray", return_value=image),
        ):
            handler._frame(query)

        build_depth.assert_called_once()
        depth_payload = build_depth.call_args.args[0]
        self.assertEqual(depth_payload["quality"], "turbo")
        self.assertEqual(depth_payload["invert"], "1")
        self.assertEqual(depth_payload["preview"], "540")
        build_blur.assert_not_called()
        render_blur.assert_not_called()
        render_depth.assert_called_once_with(
            "clip.mp4",
            1.25,
            cfg,
            max_height=360,
        )
        self.assertTrue(lock.entered)
        self.assertTrue(lock.exited)
        handler._send.assert_called_once_with(
            200,
            b"jpeg",
            "image/jpeg",
        )


if __name__ == "__main__":
    unittest.main()

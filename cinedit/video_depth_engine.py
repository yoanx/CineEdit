"""Streaming Video Depth Anything wrapper."""
from __future__ import annotations

import os
import threading
from typing import Callable

import numpy as np
import torch

from .video_depth_anything.video_depth_stream import VideoDepthAnything
from .weights import video_depth_model_dir

_PREVIEW_ENGINE: "VideoDepthEngine | None" = None
_PREVIEW_LOCK = threading.RLock()


def _device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "cuda":
            raise RuntimeError("specified cuda but CUDA is not available")
        return torch.device("cpu")
    return torch.device(pref)


class VideoDepthEngine:
    """Run the temporal model one frame at a time with resident hidden state."""

    is_video = True

    def __init__(self, device: str = "auto", model=None):
        self.device = _device(device)
        self.fp16 = self.device.type == "cuda"
        if model is None:
            model = VideoDepthAnything(
                encoder="vits",
                features=64,
                out_channels=[48, 96, 192, 384],
            )
            checkpoint = os.path.join(
                video_depth_model_dir(),
                "video_depth_anything_vits.pth",
            )
            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(state, strict=True)
            model = model.to(self.device).eval()
        self._model = model

    def estimate(
        self,
        frame: torch.Tensor,
        input_size: int,
        log: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Return one [1,1,H,W] depth map for a normalized RGB tensor."""
        rgb = (
            frame.detach()
            .to(device="cpu", dtype=torch.float32)
            .mul(255.0)
            .round()
            .clamp(0.0, 255.0)
            .to(torch.uint8)
        )
        image = rgb[0].permute(1, 2, 0).contiguous().numpy()
        depth = self._model.infer_video_depth_one(
            image,
            input_size=input_size,
            device=self.device.type,
            fp32=not self.fp16,
        )
        return (
            torch.from_numpy(np.asarray(depth))
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def reset(self) -> None:
        """Reset the stream's temporal caches before a standalone preview."""
        if self._model is None:
            return
        self._model.transform = None
        self._model.frame_id_list = []
        self._model.frame_cache_list = []
        self._model.id = -1

    def release(self) -> None:
        self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def get_preview_video_depth_engine(device: str = "auto") -> VideoDepthEngine:
    """Return a reset streaming engine for single-frame depth previews."""
    global _PREVIEW_ENGINE
    with _PREVIEW_LOCK:
        want = _device(device)
        if _PREVIEW_ENGINE is None or _PREVIEW_ENGINE.device != want:
            if _PREVIEW_ENGINE is not None:
                _PREVIEW_ENGINE.release()
            _PREVIEW_ENGINE = VideoDepthEngine(device)
        _PREVIEW_ENGINE.reset()
        return _PREVIEW_ENGINE


def release_preview_video_depth_engine() -> None:
    global _PREVIEW_ENGINE
    with _PREVIEW_LOCK:
        if _PREVIEW_ENGINE is not None:
            _PREVIEW_ENGINE.release()
            _PREVIEW_ENGINE = None

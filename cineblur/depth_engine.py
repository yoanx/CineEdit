"""Offline Depth Anything V2 inference."""
from __future__ import annotations

import logging
import threading
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .weights import depth_model_dir


logger = logging.getLogger(__name__)
_SHARED_ENGINE: "DepthEngine | None" = None
_SHARED_ENGINE_LOCK = threading.RLock()
_OOM_FALLBACKS = {
    518: (392, "fast"),
    392: (336, "turbo"),
}


def _pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "cuda":
            raise RuntimeError("specified cuda but CUDA is not available")
        return torch.device("cpu")
    return torch.device(pref)


class DepthEngine:
    """Depth Anything V2 estimator backed by local model files."""

    def __init__(self, model_dir: str | None = None, device: str = "auto"):
        self.device = _pick_device(device)
        self.fp16 = self.device.type == "cuda"
        self.last_fallback: tuple[int, int, str] | None = None
        path = model_dir or depth_model_dir()
        self._processor = AutoImageProcessor.from_pretrained(
            path,
            local_files_only=True,
            use_fast=False,
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            path,
            local_files_only=True,
        )
        self._model.to(self.device).eval()
        if self.fp16:
            self._model.half()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _estimate_at(self, frame: torch.Tensor, input_size: int) -> torch.Tensor:
        if frame.ndim != 4 or frame.shape[:2] != (1, 3):
            raise ValueError("frame must have shape [1, 3, H, W]")
        if input_size <= 0 or input_size % 14 != 0:
            raise ValueError("input_size must be a positive multiple of 14")

        height, width = frame.shape[-2:]
        image = frame.detach().to(device="cpu", dtype=torch.float32)
        image = image.mul(255.0).round().clamp_(0.0, 255.0).to(torch.uint8)
        inputs = self._processor(
            images=image,
            size={"height": input_size, "width": input_size},
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.fp16:
            pixel_values = pixel_values.half()

        predicted_depth = self._model(pixel_values=pixel_values).predicted_depth
        depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return depth.float()

    @torch.inference_mode()
    def estimate(
        self,
        frame: torch.Tensor,
        input_size: int,
        log: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Estimate depth, retrying once at the next lower preset after CUDA OOM."""
        try:
            return self._estimate_at(frame, input_size)
        except torch.cuda.OutOfMemoryError:
            fallback = _OOM_FALLBACKS.get(input_size)
            if self.device.type != "cuda" or fallback is None:
                raise
            retry_size, preset = fallback
            torch.cuda.empty_cache()
            message = (
                f"CUDA OOM at {input_size}px; retrying once at "
                f"{retry_size}px ({preset})"
            )
            logger.warning(message)
            if log:
                log(message)
            self.last_fallback = (input_size, retry_size, preset)
            return self._estimate_at(frame, retry_size)

    def release(self) -> None:
        self._model = None
        self._processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def get_depth_engine(device: str = "auto") -> DepthEngine:
    """Return the process-wide resident depth engine.

    The accessor owns the shared engine. Preview and full-render callers only
    borrow it, so a renderer must not release it when it closes.
    """
    global _SHARED_ENGINE
    want = _pick_device(device)
    with _SHARED_ENGINE_LOCK:
        if _SHARED_ENGINE is None or _SHARED_ENGINE.device != want:
            if _SHARED_ENGINE is not None:
                _SHARED_ENGINE.release()
            _SHARED_ENGINE = DepthEngine(device=want.type)
        return _SHARED_ENGINE


def release_depth_engine() -> None:
    """Release the process-wide engine, primarily for shutdown and tests."""
    global _SHARED_ENGINE
    with _SHARED_ENGINE_LOCK:
        if _SHARED_ENGINE is not None:
            _SHARED_ENGINE.release()
            _SHARED_ENGINE = None

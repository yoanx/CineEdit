"""Offline Depth Anything V2 inference."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .weights import depth_model_dir


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

    @torch.inference_mode()
    def estimate(self, frame: torch.Tensor, input_size: int) -> torch.Tensor:
        """Estimate depth for an RGB frame in the range 0..1."""
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

    def release(self) -> None:
        self._model = None
        self._processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

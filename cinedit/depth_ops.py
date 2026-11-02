from __future__ import annotations

import os

import torch


def is_processed_depth_source(src: str) -> bool:
    """Return whether a path looks like an existing motion-blur output."""
    name = os.path.basename(src.replace("\\", "/"))
    stem, _ = os.path.splitext(name)
    return stem.lower().endswith("_blur")


def depth_range(depth: torch.Tensor, sample_step: int = 8) -> tuple[float, float]:
    sample = depth.detach().float().flatten()[:: max(1, sample_step)]
    low = torch.quantile(sample, 0.01).item()
    high = torch.quantile(sample, 0.99).item()
    return float(low), float(high)


def normalize_depth(
    depth: torch.Tensor, low: float, high: float, invert: bool = False
) -> torch.Tensor:
    scale = max(high - low, 1e-6)
    out = ((depth.float() - low) / scale).clamp(0.0, 1.0)
    return 1.0 - out if invert else out


def smooth_range(
    previous: tuple[float, float] | None,
    current: tuple[float, float],
    alpha: float = 0.95,
) -> tuple[float, float]:
    """Keep depth visualization contrast from jumping between frames."""
    if previous is None:
        return current
    alpha = max(0.0, min(1.0, alpha))
    return (
        alpha * previous[0] + (1.0 - alpha) * current[0],
        alpha * previous[1] + (1.0 - alpha) * current[1],
    )

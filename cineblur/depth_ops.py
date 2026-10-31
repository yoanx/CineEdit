from __future__ import annotations

import torch


def depth_range(depth: torch.Tensor, sample_step: int = 8) -> tuple[float, float]:
    sample = depth.detach().float().flatten()[:: max(1, sample_step)]
    low = sample.min().item()
    high = sample.max().item()
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
    alpha: float = 0.85,
) -> tuple[float, float]:
    if previous is None:
        return current
    return (
        current[0] + alpha * (previous[0] - current[0]),
        current[1] + alpha * (previous[1] - current[1]),
    )

from __future__ import annotations

import torch


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


def temporal_blend(
    current: torch.Tensor,
    previous: torch.Tensor | None,
    static_mask: torch.Tensor,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Blend only pixels whose source frame is static."""
    if previous is None:
        return current
    alpha = max(0.0, min(1.0, alpha))
    blended = previous.lerp(current, 1.0 - alpha)
    return torch.where(static_mask, blended, current)

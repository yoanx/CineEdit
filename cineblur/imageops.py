"""图像算子：颜色空间、光流重采样、引导滤波、抖动图案。

设计要点
--------
1. 所有 GPU 算子都按「一次分配、反复复用」写。逐样本新建张量在 1080p 上
   每次要写 12 MB，24 个子帧就是 288 MB 的纯分配开销，实际会拖慢 20% 以上。
2. 采样网格以 NHWC 布局缓存。grid_sample 要求 [B,H,W,2]，若每次都用
   cat/permute 现场拼，光是把 BCHW 转过来就要多两次全图拷贝。
3. sRGB 转换保持精确公式。这里不做 LUT 近似 —— 输入虽然是 8 bit，但转换
   每帧只发生一次（见 pipeline 的线性光帧缓存），省下的时间抵不上精度风险。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- 颜色空间


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB (0..1) -> 线性光。物理正确的快门积分必须在线性空间做。"""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def luma(x: torch.Tensor) -> torch.Tensor:
    """Rec.709 亮度，用于引导滤波与自适应判定。"""
    return (0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3])


# ---------------------------------------------------------------- 采样网格

_GRID_CACHE: dict[tuple, torch.Tensor] = {}


def pixel_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """归一化采样基准网格 [1,2,H,W]，坐标顺序 (x, y)，align_corners=False。"""
    key = (h, w, str(device), str(dtype))
    g = _GRID_CACHE.get(key)
    if g is not None:
        return g
    xs = torch.arange(w, device=device, dtype=dtype)
    ys = torch.arange(h, device=device, dtype=dtype)
    gx = (2.0 * (xs + 0.5) / w - 1.0).view(1, 1, 1, w).expand(1, 1, h, w)
    gy = (2.0 * (ys + 0.5) / h - 1.0).view(1, 1, h, 1).expand(1, 1, h, w)
    g = torch.cat([gx, gy], dim=1).contiguous()
    _GRID_CACHE[key] = g
    return g


def grid_nhwc(h: int, w: int, device, dtype) -> torch.Tensor:
    """[1,H,W,2] 布局的基准网格，直接喂给 grid_sample 无需再 permute。"""
    key = ("nhwc", h, w, str(device), str(dtype))
    g = _GRID_CACHE.get(key)
    if g is not None:
        return g
    g = pixel_grid(h, w, device, dtype).permute(0, 2, 3, 1).contiguous()
    _GRID_CACHE[key] = g
    return g


def flow_nhwc(flow_px: torch.Tensor) -> torch.Tensor:
    """像素位移场 [1,2,H,W] -> 归一化位移 [1,H,W,2]。

    归一化只要一次乘除，却避免了每个子帧重复 cat + permute。
    """
    h, w = flow_px.shape[-2:]
    out = torch.empty(1, h, w, 2, device=flow_px.device, dtype=flow_px.dtype)
    out[..., 0] = flow_px[:, 0] * (2.0 / w)
    out[..., 1] = flow_px[:, 1] * (2.0 / h)
    return out


def warp(img: torch.Tensor, flow_px: torch.Tensor, mode: str = "bilinear",
         padding: str = "border") -> torch.Tensor:
    """按像素位移场重采样图像。

    img: [B,C,H,W]；flow_px: [B,2,H,W]，单位像素，(dx, dy)，源坐标 = 目标坐标 + flow。
    冷路径用（一致性权重、散射通路），热路径请用 Warper。
    """
    b, _, h, w = img.shape
    if flow_px.shape[0] != b:
        flow_px = flow_px.expand(b, -1, -1, -1)
    g = flow_nhwc(flow_px)
    g.add_(grid_nhwc(h, w, img.device, g.dtype))
    return F.grid_sample(img, g, mode=mode, padding_mode=padding, align_corners=False)


class Warper:
    """热路径重采样器：复用网格与位移缓冲，逐子帧零分配。

    每调用一次只有 1 次乘法 + 1 次加法作用在 [1,H,W,2] 上，再交给 grid_sample。
    相比「每样本重建网格」，1080p 下可省约 40% 的显存带宽。
    """

    __slots__ = ("h", "w", "grid", "_off", "_mul", "dtype", "device")

    def __init__(self, h: int, w: int, device, dtype, mode: str = "bilinear"):
        self.h, self.w, self.dtype, self.device = h, w, dtype, device
        self.grid = grid_nhwc(h, w, device, dtype)
        self._off = torch.empty(1, h, w, 2, device=device, dtype=dtype)
        self._mul = (mode,)

    def __call__(self, img_l: torch.Tensor, field_nhwc: torch.Tensor, scale,
                mode: str = "bilinear") -> torch.Tensor:
        """img_l @ 由 field（_nhwc 归一化位移）× scale 指定的源坐标。"""
        if isinstance(scale, torch.Tensor):
            torch.mul(field_nhwc, scale, out=self._off)
        else:
            if scale == 1.0:
                self._off.copy_(field_nhwc)
            elif scale == 0.0:
                self._off.zero_()
            else:
                torch.mul(field_nhwc, scale, out=self._off)
        torch.add(self.grid, self._off, out=self._off)
        return F.grid_sample(img_l, self._off, mode=mode, padding_mode="border",
                             align_corners=False)


# ---------------------------------------------------------------- 抖动图案

_BLUE_CACHE: dict[tuple, torch.Tensor] = {}
_TILE_N = 256


def _blue_tile(seed: int) -> torch.Tensor:
    """256×256 蓝噪声瓦片，值域 [0,1)。

    做法：白噪声减去低通分量（高通），再按幅值排序归一化。空间上相邻的像素
    因此倾向于落在不同采样区间 —— 低采样数下拖影边界不再是「团块状噪点」，
    而是均匀铺开。对固定 seed 恒定，不引入时域闪烁。
    """
    t = _BLUE_CACHE.get(seed)
    if t is not None:
        return t
    g = torch.Generator()
    g.manual_seed(seed * 7919 + 104729)
    base = torch.rand(1, 1, _TILE_N, _TILE_N, generator=g)
    kernel = torch.ones(1, 1, 7, 7) / 49.0
    low = F.conv2d(F.pad(base, (3, 3, 3, 3), mode="replicate"), kernel)
    hp = (base - low).reshape(-1)
    rank = hp.argsort().argsort().to(torch.float32)
    rank /= (_TILE_N * _TILE_N - 1.0)
    t = rank.reshape(1, 1, _TILE_N, _TILE_N)
    _BLUE_CACHE[seed] = t
    return t


def jitter_field(n: int, h: int, w: int, device, dtype, seed: int = 0) -> torch.Tensor:
    """[N,1,H,W] 抖动偏移量，每个子帧一片。

    一次性构建并缓存：渲染循环里只做切片，不再分配。
    """
    key = ("jit", n, h, w, str(device), str(dtype), seed)
    t = _BLUE_CACHE.get(key)
    if t is not None:
        return t
    reph, repw = math.ceil(h / _TILE_N), math.ceil(w / _TILE_N)
    outs = []
    for j in range(n):
        tile = _blue_tile(seed + j)
        # 逐样本整块平移，打散瓦片周期，避免 256px 的重复纹理
        dy = (j * 61 + seed * 17) % _TILE_N
        dx = (j * 113 + seed * 29) % _TILE_N
        tile = torch.roll(tile, shifts=(dy, dx), dims=(-2, -1))
        outs.append(tile.repeat(1, 1, reph, repw)[:, :, :h, :w])
    t = torch.cat(outs, dim=0).to(device=device, dtype=dtype)
    _BLUE_CACHE[key] = t
    return t


# ---------------------------------------------------------------- 引导滤波


def _box(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2,
                        count_include_pad=False)


def guided_filter(guide: torch.Tensor, src: torch.Tensor, radius: int = 3,
                  eps: float = 1e-3) -> torch.Tensor:
    """以 guide 为边缘引导，对 src 做边缘保持平滑。

    guide 会被折成单通道亮度：联合引导滤波的系数 a、b 对每个 src 通道各一组，
    但均值/方差只依赖引导图。用 3 通道原图做引导要多算两倍盒滤波，视觉收益
    可以忽略。方差从通道循环里提出来，避免重复计算。
    """
    if guide.shape[1] != 1:
        guide = luma(guide)
    k = 2 * radius + 1
    mean_i = _box(guide, k)                       # [1,1,H,W]
    var_i = (_box(guide * guide, k) - mean_i * mean_i).clamp_min(0.0)
    inv = 1.0 / (var_i + eps)

    outs = []
    for c in range(src.shape[1]):
        p = src[:, c:c + 1]
        mean_p = _box(p, k)
        cov = _box(guide * p, k) - mean_i * mean_p
        a = cov * inv
        b = mean_p - a * mean_i
        outs.append(_box(a, k) * guide + _box(b, k))
    return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


def upsample_flow(flow_lr: torch.Tensor, flow_scale: float,
                  guide: torch.Tensor | None = None,
                  radius: int = 3) -> torch.Tensor:
    """把低分辨率光流放大到引导图（原帧）分辨率。

    位移以低分辨率像素为单位，放大后需乘 1/flow_scale 换算成原分辨率像素。
    """
    h, w = guide.shape[-2:] if guide is not None else flow_lr.shape[-2:]
    if flow_lr.shape[-2:] != (h, w):
        flow = F.interpolate(flow_lr, size=(h, w), mode="bilinear", align_corners=False)
    else:
        flow = flow_lr
    flow = flow * (1.0 / flow_scale)
    if guide is not None and radius > 0:
        flow = guided_filter(guide, flow, radius=radius, eps=1e-4)
    return flow.contiguous()

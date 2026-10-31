"""单帧预览：抽取目标时刻附近的三帧，跑一次完整的模糊内核。

用于界面里拖动快门角时的实时反馈 —— 复杂镜头靠猜参数是不可行的。

两级缓存
--------
1. **解码缓存**：同一帧位置的原始帧（ffmpeg 起进程 + 定位 + 缩放，约 150~250 ms）
2. **光流缓存**：只依赖帧内容与光流档位，与快门角/采样数无关

因此拖动快门角、采样数、抗伪影这些"纯积分参数"时，预览只需要重跑快门积分
（540p 约 15~25 ms），不再重复解码与光流。这是交互手感的关键。
"""
from __future__ import annotations

import functools
import os
import subprocess
import threading

import numpy as np
import torch

from .blur import render_frame
from .config import BlurConfig, DepthConfig
from .depth_engine import DepthEngine
from .depth_ops import depth_range, normalize_depth
from .flow import FlowEngine
from .imageops import srgb_to_linear
from .media import CREATIONFLAGS, find_exe, probe

_ENGINE: FlowEngine | None = None
_LOCK = threading.Lock()
_STATE: dict = {"key": None}
_ORIGINAL_STATE: dict = {"key": None}
_DEPTH_ENGINE: DepthEngine | None = None
_DEPTH_LOCK = threading.Lock()
_DEPTH_FRAME_CACHE: dict[tuple, np.ndarray] = {}
_DEPTH_OUTPUT_CACHE: dict[tuple, np.ndarray] = {}


def get_engine(cfg: BlurConfig) -> FlowEngine:
    """复用一个引擎实例，避免每次预览都重新加载 RAFT 权重。"""
    global _ENGINE
    with _LOCK:
        want = ("cuda" if cfg.device != "cpu" else "cpu")
        if (_ENGINE is None or _ENGINE.model_name != cfg.flow_model
                or _ENGINE.device.type != want):
            if _ENGINE is not None:
                _ENGINE.release()
            _ENGINE = FlowEngine(model=cfg.flow_model, iters=cfg.flow_iters,
                                 device=cfg.device, scale=cfg.flow_scale,
                                 early_exit=cfg.early_exit, fp16=cfg.flow_fp16)
        else:
            _ENGINE.iters = cfg.flow_iters
            _ENGINE.scale = cfg.flow_scale
            _ENGINE.early_exit = cfg.early_exit
            _ENGINE.fp16 = cfg.flow_fp16
        return _ENGINE


def _extract_frames(src: str, first: int, w: int, h: int, count: int = 3):
    """从第 first 帧起抽取 count 帧 rgb24。"""
    ffmpeg = find_exe("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg")
    info = probe(src)
    fps = float(info.fps)
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
           "-ss", f"{max(0.0, first / fps):.4f}", "-i", src,
           "-frames:v", str(count),
           "-vf", f"scale={w}:{h}:flags=lanczos,format=rgb24",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True, creationflags=CREATIONFLAGS)
    buf = out.stdout
    n = len(buf) // (w * h * 3)
    if n == 0:
        raise RuntimeError("无法解码该时间点")
    # copy() 让数组可写：torch.from_numpy 不接受只读缓冲区
    return np.frombuffer(buf[:n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3).copy()


@functools.lru_cache(maxsize=4)
def _size_of(src: str, mtime: float) -> tuple[int, int]:
    i = probe(src)
    w, h = i.width, i.height
    return (w - w % 2, h - h % 2)


@functools.lru_cache(maxsize=4)
def _fps_of(src: str, mtime: float) -> float:
    return float(probe(src).fps)


def _resolve_size(src: str, max_height: int) -> tuple[int, int]:
    w, h = _size_of(src, os.path.getmtime(src))
    if max_height and h > max_height:
        w2 = int(round(w * max_height / h))
        w, h = w2 - w2 % 2, max_height - max_height % 2
    return w, h


def _dev(cfg: BlurConfig) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")


def _to_t(a: np.ndarray, dev) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None] \
        .float().to(dev) / 255.0


def _cache_put(cache: dict[tuple, np.ndarray], key: tuple, value: np.ndarray) -> None:
    if len(cache) >= 8 and key not in cache:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _depth_engine(cfg: DepthConfig) -> DepthEngine:
    global _DEPTH_ENGINE
    with _DEPTH_LOCK:
        want = _dev(cfg)
        if _DEPTH_ENGINE is None or _DEPTH_ENGINE.device != want:
            if _DEPTH_ENGINE is not None:
                _DEPTH_ENGINE.release()
            _DEPTH_ENGINE = DepthEngine(device=want.type)
        return _DEPTH_ENGINE


def _depth_frame(src: str, idx: int, w: int, h: int) -> np.ndarray:
    key = (src, os.path.getmtime(src), idx, w, h)
    frame = _DEPTH_FRAME_CACHE.get(key)
    if frame is None:
        frame = _extract_frames(src, idx, w, h, 1)[0].copy()
        _cache_put(_DEPTH_FRAME_CACHE, key, frame)
    return frame


def _frames_at(src: str, frame_idx: int, w: int, h: int, dev):
    """取 frame_idx 及其前后帧，带解码缓存。返回 (prev, cur, next, 是否首帧)。"""
    key = ("dec", src, os.path.getmtime(src), frame_idx, w, h)
    if _STATE.get("key") == key:
        return _STATE["frames"]
    start = max(0, frame_idx - 1)
    arrs = _extract_frames(src, start, w, h, 3)
    k = frame_idx - start                       # 窗口内的中心帧下标
    if k >= len(arrs):
        k = len(arrs) - 1
    prev_a = arrs[max(0, k - 1)]
    cur_a = arrs[k]
    next_a = arrs[min(len(arrs) - 1, k + 1)]
    trio = (_to_t(prev_a, dev), _to_t(cur_a, dev), _to_t(next_a, dev))
    _STATE["key"] = key
    _STATE["frames"] = trio
    _STATE["flow_key"] = None
    return trio


def _flows(src: str, frame_idx: int, w: int, h: int, cfg: BlurConfig, dev, trio):
    """光流带缓存：只依赖帧内容与光流档位。"""
    fkey = ("flow", src, frame_idx, w, h, cfg.flow_model, cfg.flow_scale,
            cfg.flow_iters, cfg.flow_dirs, cfg.early_exit, cfg.flow_fp16)
    if _STATE.get("flow_key") == fkey:
        return _STATE["flows"]
    eng = get_engine(cfg)
    eng.drop_cache()
    prev_t, cur_t, next_t = trio
    f_nf = eng.flow(cur_t, next_t, 0, 1)
    both = cfg.flow_dirs == "both"
    f_nb = eng.flow(next_t, cur_t, 1, 0) if both else None
    f_pf = eng.flow(prev_t, cur_t, 0, 1) if both else None
    val = (f_nf, f_nb, f_pf)
    _STATE["flow_key"] = fkey
    _STATE["flows"] = val
    return val


def render_preview(src: str, t: float, cfg: BlurConfig,
                   max_height: int = 720) -> np.ndarray:
    """返回 [H,W,3] uint8 的预览帧（已经过运动模糊）。"""
    w, h = _resolve_size(src, max_height)
    fps = _fps_of(src, os.path.getmtime(src))
    idx = int(round(t * fps))
    dev = _dev(cfg)
    trio = _frames_at(src, idx, w, h, dev)
    f_nf, f_nb, f_pf = _flows(src, idx, w, h, cfg, dev, trio)

    prev_t, cur_t, next_t = trio
    if cfg.linear_light:
        lin = (srgb_to_linear(prev_t), srgb_to_linear(cur_t), srgb_to_linear(next_t))
    else:
        lin = (prev_t, cur_t, next_t)
    out = render_frame(prev_t, cur_t, next_t, f_nf, -f_nf, cfg,
                       f_nb=f_nb, f_pf=f_pf, linear=lin)
    return (out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.inference_mode()
def render_depth_preview(src: str, t: float, cfg: DepthConfig,
                         max_height: int = 540) -> np.ndarray:
    """Return a cached RGB visualization of the estimated depth frame."""
    mtime = os.path.getmtime(src)
    w, h = _resolve_size(src, max_height)
    fps = _fps_of(src, mtime)
    idx = int(round(t * fps))
    out_key = (src, mtime, idx, cfg.quality, cfg.invert, max_height)
    cached = _DEPTH_OUTPUT_CACHE.get(out_key)
    if cached is not None:
        return cached

    frame = _depth_frame(src, idx, w, h)
    engine = _depth_engine(cfg)
    depth = engine.estimate(_to_t(frame, engine.device), cfg.input_size())
    low, high = depth_range(depth)
    gray = normalize_depth(depth, low, high, cfg.invert)
    gray_u8 = (
        gray.clamp(0, 1)[0, 0]
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    out = np.repeat(gray_u8[..., None], 3, axis=2)
    _cache_put(_DEPTH_OUTPUT_CACHE, out_key, out)
    return out


def render_original(src: str, t: float, max_height: int = 720) -> np.ndarray:
    w, h = _resolve_size(src, max_height)
    fps = _fps_of(src, os.path.getmtime(src))
    idx = int(round(t * fps))
    dec_key = ("dec", src, os.path.getmtime(src), idx, w, h)
    if _STATE.get("key") == dec_key:
        cur = _STATE["frames"][1]
        return (cur[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    orig_key = ("orig", src, os.path.getmtime(src), idx, w, h)
    if _ORIGINAL_STATE.get("key") == orig_key:
        return _ORIGINAL_STATE["frame"]

    frames = _extract_frames(src, idx, w, h, 1)
    frame = frames[0].copy()
    _ORIGINAL_STATE["key"] = orig_key
    _ORIGINAL_STATE["frame"] = frame
    return frame


def warmup(cfg: BlurConfig) -> None:
    """预加载权重并跑一次小尺寸前向。

    只加载权重是不够的：cuDNN 会对每个新尺寸做算子自动调优，第一次前向仍要
    额外数秒。这里用 288x512 的空图跑通整条路径，把这段开销挪到启动阶段。
    """
    try:
        eng = get_engine(cfg)
        eng.net
        dummy = torch.zeros(1, 3, 288, 512, device=eng.device)
        eng.flow(dummy, dummy, None, None)
        torch.cuda.synchronize() if eng.device.type == "cuda" else None
    except Exception:  # noqa: BLE001
        pass

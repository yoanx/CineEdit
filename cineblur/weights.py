"""随包分发的 RAFT 权重：首次运行前放进 torch 缓存。

torchvision 的默认行为是按权重文件的哈希名去 ``torch.hub`` 缓存里找，找不到就
从 download.pytorch.org 下载。别人的机器上没有这个缓存，而该域名在国内经常只有
几十 KB/s——25 MB 能卡十几分钟，表现出来就是"双击后一直黑屏没反应"。

所以把权重打进安装包，模型加载前先检查缓存、缺失就直接复制过去。
"""
from __future__ import annotations

import os
import shutil
import sys

# 与 torchvision 内部使用的文件名一致（含哈希后缀，不能改名）
WEIGHT_FILES = (
    "raft_large_C_T_SKHT_V2-ff5fadd5.pth",
    "raft_small_C_T_V2-01064c6d.pth",
)
DEPTH_MODEL_DIR = "depth_anything_v2_small"


def _bundle_dir() -> str | None:
    """按优先级查找随包的权重目录。"""
    roots: list[str] = []
    if getattr(sys, "frozen", False):
        roots.append(os.path.dirname(os.path.abspath(sys.executable)))
    if hasattr(sys, "_MEIPASS"):
        roots.append(sys._MEIPASS)
    here = os.path.dirname(os.path.abspath(__file__))
    roots += [here, os.path.dirname(here), os.getcwd()]
    for r in roots:
        for parts in (("assets", "weights"), ("weights",)):
            d = os.path.join(r, *parts)
            if os.path.isdir(d):
                return d
    return None


def cache_dir() -> str:
    """torch 的权重缓存目录（与 torchvision 下载时用的是同一个）。"""
    try:
        import torch
        return os.path.join(torch.hub.get_dir(), "checkpoints")
    except Exception:  # noqa: BLE001
        return os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")


def depth_model_dir() -> str:
    """返回随包的 Depth Anything V2 模型目录。"""
    src = _bundle_dir()
    if not src:
        raise FileNotFoundError("depth model directory not found")
    path = os.path.join(src, DEPTH_MODEL_DIR)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"depth model not found: {path}")
    return path


def depth_model_available() -> bool:
    """Return whether the required local depth model files are present."""
    try:
        path = depth_model_dir()
    except (FileNotFoundError, OSError):
        return False
    config = os.path.join(path, "config.json")
    processor = os.path.join(path, "preprocessor_config.json")
    weights = (
        os.path.join(path, "model.safetensors"),
        os.path.join(path, "pytorch_model.bin"),
    )
    return (
        os.path.isfile(config)
        and os.path.isfile(processor)
        and any(os.path.isfile(item) for item in weights)
    )


def ensure_weights() -> int:
    """把缺失的权重复制进缓存，返回复制的文件数。

    已存在且大小一致的直接跳过——不覆盖用户自己下载的版本。
    """
    src = _bundle_dir()
    if not src:
        return 0
    dst = cache_dir()
    copied = 0
    for name in WEIGHT_FILES:
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if not os.path.isfile(s):
            continue
        try:
            if os.path.isfile(d) and os.path.getsize(d) == os.path.getsize(s):
                continue
            os.makedirs(dst, exist_ok=True)
            tmp = d + ".part"
            shutil.copyfile(s, tmp)
            os.replace(tmp, d)
            copied += 1
        except OSError:
            # 缓存目录不可写时不要阻断主流程，退化为让 torchvision 自己下载
            pass
    return copied

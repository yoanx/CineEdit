"""运行配置。所有可调参数集中在此，CLI / 界面共享。"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class BlurConfig:
    # ---- 快门 ----
    angle: float = 180.0          # 快门角（度）。360=整帧曝光，180=电影标准，90=硬朗
    samples: int = 16             # 快门区间内的子帧采样数（越大越平滑）
    strength: float = 1.0         # 轨迹长度缩放（>1 会被截断到单帧曝光上限）
    occlusion: float = 0.0        # 前后向一致性抑制（需 flow_dirs="both"）
    photometric: float = 0.0      # 光度一致性抑制 0..1（真实素材抗伪影用 0.3~0.5）
    photo_sigma: float = 0.4      # 光度一致性容差（0..1 色阶）
    jitter: bool = True           # 蓝噪声逐像素抖动，消除阶梯断层
    linear_light: bool = True     # 线性光空间积分（物理正确）
    bicubic: bool = False         # 双三次重采样（拖影更柔，约慢 15%）
    disocclusion: float = 0.0     # 【实验性】散射回填强度，实测不优于纯 gather，默认关
    splat_samples: int = 6        # 散射通路采样数

    # ---- 光流 ----
    flow_model: str = "raft_large"   # raft_large | raft_small
    flow_scale: float = 0.5          # 光流计算分辨率比例
    flow_iters: int = 8              # RAFT 迭代上限（收敛即提前退出）
    flow_dirs: str = "single"        # single=1 次/帧（快 2 倍），both=双向一致性（更稳）
    early_exit: bool = True          # 收敛提前退出
    flow_fp16: bool = False          # RAFT 用半精度（更快，长位移偶尔掉精度）
    cons_sigma: float = 1.5          # 前后向一致性容差（像素）

    # ---- 设备 ----
    device: str = "auto"          # auto | cuda | cpu

    # ---- 输出 ----
    codec: str = "auto"           # auto | h264_nvenc | hevc_nvenc | libx264 | libx265
    crf: int = 16                 # 恒定质量因子
    preset: str = ""              # 留空则用编码器默认高质量档

    # ---- 预览 ----
    preview_height: int = 0       # >0 时先按该高度输出（低分辨率快速预览）

    # ---- 裁剪（秒，0 表示从头/到尾）----
    trim_start: float = 0.0
    trim_end: float = 0.0

    def shutter_fraction(self) -> float:
        """一个帧间隔内快门开启的比例。180° @ 任意 fps 均为 0.5。"""
        return max(0.0, min(1.0, self.angle / 360.0))

    def validate(self) -> None:
        if self.samples < 2:
            raise ValueError("samples 至少为 2")
        if self.samples > 256:
            raise ValueError("samples 上限 256")
        if not 0 <= self.angle <= 360:
            raise ValueError("angle 必须在 0..360 之间")
        if not 0 < self.flow_scale <= 1.0:
            raise ValueError("flow_scale 必须在 (0, 1] 之间")
        if self.flow_model not in ("raft_large", "raft_small"):
            raise ValueError("flow_model 只支持 raft_large / raft_small")
        if self.flow_dirs not in ("single", "both"):
            raise ValueError("flow_dirs 只支持 single / both")


@dataclass
class DepthConfig:
    quality: str = "fast"
    output_height: int = 0
    invert: bool = False
    codec: str = "auto"
    crf: int = 18
    device: str = "auto"
    trim_start: float = 0.0
    trim_end: float = 0.0

    def input_size(self) -> int:
        return {"turbo": 336, "fast": 392, "fine": 518}[self.quality]

    def output_name(self, src: str) -> str:
        stem, _ = os.path.splitext(src)
        return stem + "_depth.mp4"

    def validate(self) -> None:
        if self.quality not in ("turbo", "fast", "fine"):
            raise ValueError("quality must be turbo, fast, or fine")
        if self.codec not in ("auto", "h264_nvenc", "libx264"):
            raise ValueError("unsupported depth codec")

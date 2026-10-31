"""量化评估：构造有解析真值的合成场景，对比本方案与 RSMB 式线性基线。

真值 (GT) 由 128 倍时间超采样积分得到，物理上等价于真实相机在一次曝光内的积分。
对比项：
  1. 原始帧（无模糊）
  2. linear  —— RSMB 式：单一方向拉伸 + gamma 空间平均 + 无遮挡处理
  3. flow(occlusion=0) —— 有子帧采样但无遮挡抑制
  4. flow(occlusion=1) —— 完整方案

场景包含三个难点：匀速平移的遮挡物（测拖尾光晕）、旋转杆（测弯曲轨迹）、
加速滑块（测非匀速位移）。
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import torch
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cineblur.config import BlurConfig
from cineblur.imageops import linear_to_srgb, srgb_to_linear
from cineblur.media import Decoder, Encoder, find_exe
from cineblur.pipeline import MotionBlurRenderer

W, H = 640, 360
FPS = 30
NFRAMES = 12
GT_SAMPLES = 128
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def _over(img, color, mask):
    m = mask[..., None]
    return img * (1 - m) + np.array(color, dtype=np.float32) * m


def scene(t: float) -> np.ndarray:
    """场景在时间 t（帧为单位）的线性光图像 [H,W,3]。"""
    x, y = _xx, _yy
    bg = (0.42
          + 0.16 * np.sin(x * 0.35 + 1.2) * np.cos(y * 0.28)
          + 0.10 * np.sin((x + y) * 0.85)
          + 0.16 * np.sign(np.sin(x * 0.11)))
    img = np.repeat(bg[..., None], 3, axis=2).astype(np.float32)

    # 加速滑块（右下）
    ax = 70 + 1.0 * t + 0.42 * t * t
    ay = 150 + 0.10 * t
    m = _smoothstep(22, 21, np.hypot(x - ax, y - ay))
    img = _over(img, (0.10, 0.62, 0.28), m)

    # 旋转杆（弯曲轨迹）
    cx, cy = 320.0, 250.0
    th = 0.22 * t
    dx, dy = x - cx, y - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    m = _smoothstep(1.0, 0.0, np.abs(v) - 11) * _smoothstep(0.0, 1.0, 118 - np.abs(u))
    img = _over(img, (0.92, 0.86, 0.20), m)

    # 匀速平移的遮挡物（测拖尾光晕）
    dxc = 60 + 19.0 * t
    m = _smoothstep(1.0, 0.0, np.hypot(x - dxc, y - 92) - 30)
    img = _over(img, (0.10, 0.22, 0.85), m)
    return img


def gt_frame(i: int) -> np.ndarray:
    """帧 i 的真值：快门区间 [i-0.5, i+0.5] 内的积分（线性光）。"""
    acc = np.zeros((H, W, 3), dtype=np.float64)
    for k in range(GT_SAMPLES):
        t = i - 0.5 + (k + 0.5) / GT_SAMPLES
        acc += scene(t)
    acc /= GT_SAMPLES
    return linear_to_srgb(torch.from_numpy(acc.astype(np.float32))).numpy()


def write_video(path: str, frames_iter):
    enc = Encoder(path, W, H, Fraction(FPS, 1), codec="libx264", crf=10)
    n = 0
    for f in frames_iter:
        enc.write((np.clip(f, 0, 1) * 255).round().astype(np.uint8))
        n += 1
    ok, err = enc.close()
    if not ok:
        raise RuntimeError(err[-500:])
    return n


def read_video(path: str):
    with Decoder(path, W, H) as dec:
        out = []
        while True:
            f = dec.read()
            if f is None:
                break
            out.append(f.astype(np.float32) / 255.0)
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def psnr_masked(a: np.ndarray, b: np.ndarray, src: np.ndarray, thr: float = 0.02) -> float:
    """只在"被模糊影响"的区域统计：|真值 - 原始帧| 显著的像素。"""
    m = np.abs(b - src).max(axis=2) > thr
    if int(m.sum()) < 100:
        return psnr(a, b)
    d = (a - b)[m].astype(np.float64)
    mse = float(np.mean(d ** 2))
    return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def main():
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(OUT, "synth_src.mp4")
    gt = os.path.join(OUT, "synth_gt.mp4")
    print("生成合成素材…")
    write_video(src, (linear_to_srgb(torch.from_numpy(scene(i))).numpy()
                      for i in range(NFRAMES)))
    write_video(gt, (gt_frame(i) for i in range(NFRAMES)))

    gt_frames = read_video(gt)
    src_frames = read_video(src)
    print(f"素材 {len(gt_frames)} 帧  {W}x{H}")

    base = dict(angle=360, samples=12, flow_scale=1.0, flow_iters=16)
    cases = [
        ("linear", BlurConfig(**base), "linear"),
        ("flow-nofilter", BlurConfig(**base, photometric=0.0, occlusion=0.0), "flow"),
        ("flow-photo", BlurConfig(**base, photometric=1.0), "flow"),
        ("flow-both", BlurConfig(**base, photometric=1.0, flow_dirs="both"), "flow"),
    ]

    results = {}
    for name, cfg, engine in cases:
        dst = os.path.join(OUT, f"synth_{name}.mp4")
        print(f"渲染 {name} …")
        r = MotionBlurRenderer(cfg)
        _render_with_engine(r, src, dst, engine)
        r.close()
        frames = read_video(dst)
        idx = range(2, min(len(frames), len(gt_frames)) - 2)
        p_all = float(np.mean([psnr(frames[k], gt_frames[k]) for k in idx]))
        # 运动区域（拖影所在）单独统计，更能反映模糊本身的质量
        p_mov = float(np.mean([psnr_masked(frames[k], gt_frames[k], src_frames[k])
                               for k in idx]))
        results[name] = (p_all, p_mov)
        print(f"   PSNR {p_all:.2f} dB   运动区 PSNR {p_mov:.2f} dB")

    print("\n结果（dB，越高越接近真实相机积分）")
    print(f"{'方案':<16}{'全图':>8}{'运动区':>10}")
    for k, (a, m) in results.items():
        print(f"{k:<16}{a:>8.2f}{m:>10.2f}")

    # 对比图：GT / 线性基线 / 本方案（单流）/ 本方案（双向）
    make_sheet([os.path.join(OUT, "synth_gt.mp4"),
                os.path.join(OUT, "synth_linear.mp4"),
                os.path.join(OUT, "synth_flow-photo.mp4"),
                os.path.join(OUT, "synth_flow-both.mp4")],
               os.path.join(OUT, "compare.png"))
    print(f"\n对比图 -> {os.path.join(OUT, 'compare.png')}")


def _render_with_engine(renderer: MotionBlurRenderer, src: str, dst: str, engine: str):
    """复用 pipeline 的渲染循环，但把 engine 参数注入单帧渲染。"""
    import cineblur.pipeline as pl
    orig = pl.render_frame

    def patched(*a, **kw):
        return orig(*a, engine=engine, **kw)

    pl.render_frame = patched
    try:
        renderer.render(src, dst, log=lambda s: None)
    finally:
        pl.render_frame = orig


def make_sheet(paths: list[str], out_png: str, frame: int = 6):
    ff = find_exe("ffmpeg")
    crop = "crop=480:230:90:40,scale=960:460:flags=neighbor"
    cmd = [ff, "-hide_banner", "-v", "error", "-y"]
    for p in paths:
        cmd += ["-i", p]
    parts, labels = [], []
    for i in range(len(paths)):
        parts.append(f"[{i}:v]{crop},trim=start_frame={frame}:end_frame={frame + 1},"
                     f"setpts=PTS-STARTPTS[p{i}]")
        labels.append(f"[p{i}]")
    fc = ";".join(parts) + ";" + "".join(labels) + f"hstack=inputs={len(paths)}"
    cmd += ["-filter_complex", fc, "-frames:v", "1", out_png]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

"""性能 / 质量权衡扫描。

用途：用同一段有解析真值的合成场景，量化「光流分辨率」与「迭代次数」这两个
最大的成本项对输出精度和耗时的影响，从而为默认档位提供依据。

输出两张表：
  1. 精度（运动区 PSNR，dB）—— 只依赖算法本身，在 640×360 上评估
  2. 耗时（ms/帧，1080p）—— 分段测量光流与积分

    python tools/tradeoff.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev                                    # noqa: E402
from cinedit.blur import render_frame                   # noqa: E402
from cinedit.config import BlurConfig                   # noqa: E402
from cinedit.flow import FlowEngine                     # noqa: E402
from cinedit.imageops import srgb_to_linear             # noqa: E402
from cinedit.pipeline import MotionBlurRenderer         # noqa: E402

OUT = ev.OUT
SCALES = [0.35, 0.5, 0.7, 1.0]
ITERS = [4, 6, 8, 12, 16]
BENCH_1080 = os.path.join(OUT, "bench1080.mp4")


def quality_table(src, gt_path):
    gts = ev.read_video(gt_path)
    srcs = ev.read_video(src)
    rows = []
    for scale in SCALES:
        for it in ITERS:
            cfg = BlurConfig(angle=360, samples=16, flow_scale=scale, flow_iters=it)
            dst = os.path.join(OUT, "_sweep.mp4")
            r = MotionBlurRenderer(cfg)
            r.render(src, dst, log=lambda s: None)
            r.close()
            frames = ev.read_video(dst)
            idx = range(2, len(gts) - 2)
            p_mov = float(np.mean([ev.psnr_masked(frames[k], gts[k], srcs[k]) for k in idx]))
            rows.append((scale, it, p_mov))
            print(f"  scale={scale:<5} iters={it:<3} 运动区 PSNR {p_mov:6.2f} dB")
    return rows


def timing_table():
    from cinedit.media import Decoder
    with Decoder(BENCH_1080, 1920, 1080) as d:
        a, b, c = d.read(), d.read(), d.read()

    def T(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None] \
            .float().cuda() / 255.0

    A, B, C = T(a), T(b), T(c)
    lin = (srgb_to_linear(A), srgb_to_linear(B), srgb_to_linear(C))

    def bench(fn, n=6):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t) / n * 1000

    rows = []
    for scale in SCALES:
        for it in ITERS:
            eng = FlowEngine("raft_large", it, "cuda", scale, early_exit=False)
            try:
                ms = bench(lambda: eng.flow(A, B, 0, 1))
            except RuntimeError:
                ms = float("nan")
            eng.release()
            rows.append((scale, it, ms))
            print(f"  1080p flow scale={scale:<5} iters={it:<3} {ms:7.1f} ms")
    cfg = BlurConfig(angle=180, samples=16)
    zero = torch.zeros(1, 2, 1080, 1920, device="cuda")
    ms_blur = bench(lambda: render_frame(A, B, C, zero, None, cfg, linear=lin))
    print(f"  1080p 积分 samples=16 {ms_blur:7.1f} ms")
    return rows, ms_blur


def main():
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(OUT, "synth_src.mp4")
    gt = os.path.join(OUT, "synth_gt.mp4")
    if not os.path.exists(gt):
        ev.main()
    print("\n[1/2] 精度扫描（640x360，运动区 PSNR）")
    q = quality_table(src, gt)
    print("\n[2/2] 1080p 光流耗时")
    t, ms_blur = timing_table()

    print("\n=== 精度矩阵（行=光流比例，列=迭代）===")
    print("         " + "".join(f"{i:>9}" for i in ITERS))
    for scale in SCALES:
        cells = [f"{p:9.2f}" for s, i, p in q if s == scale]
        print(f"{scale:<9}" + "".join(cells))
    print("\n=== 1080p 光流耗时 ms（行=比例，列=迭代）===")
    print("         " + "".join(f"{i:>9}" for i in ITERS))
    for scale in SCALES:
        cells = [f"{m:9.1f}" for s, i, m in t if s == scale]
        print(f"{scale:<9}" + "".join(cells))
    print(f"\n积分阶段（samples=16）{ms_blur:.1f} ms")


if __name__ == "__main__":
    main()

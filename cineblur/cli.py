"""命令行入口。

用法示例
--------
    python -m cineblur in.mp4 -o out.mp4 --angle 180 --samples 16
    python -m cineblur in.mp4 -o out.mp4 --fast            # 快速档
    python -m cineblur in.mp4 -o out.mp4 --engine linear   # RSMB 式基线，用于对比
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .config import BlurConfig
from .media import find_exe, probe
from .pipeline import MotionBlurRenderer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cineblur",
        description="CineBlur —— 基于光流子帧采样的物理级视频运动模糊",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="输入视频")
    p.add_argument("-o", "--output", help="输出视频（默认 <输入>_blur.mp4）")

    g = p.add_argument_group("快门")
    g.add_argument("--angle", type=float, default=180.0,
                   help="快门角（度）：180 电影标准，360 整帧曝光，90 硬朗")
    g.add_argument("--samples", type=int, default=16, help="快门内子帧采样数")
    g.add_argument("--strength", type=float, default=1.0, help="轨迹长度缩放")
    g.add_argument("--occlusion", type=float, default=0.0,
                   help="前后向光流一致性抑制 0..1，需配合 --flow-dirs both")
    g.add_argument("--photometric", type=float, default=0.0,
                   help="光度一致性抗伪影 0..1，真实素材建议 0.3~0.5")
    g.add_argument("--robust", action="store_true",
                   help="稳健模式：双向光流 + 双重一致性抑制，伪影最少但慢一倍")
    g.add_argument("--no-jitter", action="store_true", help="关闭逐像素抖动（会出现阶梯断层）")
    g.add_argument("--gamma-space", action="store_true",
                   help="在 gamma 空间积分（不推荐，仅用于对照 RSMB 行为）")
    g.add_argument("--bicubic", action="store_true",
                   help="双三次重采样，拖影更柔，约慢 15%%")

    g = p.add_argument_group("光流")
    g.add_argument("--flow-model", default="raft_large", choices=["raft_large", "raft_small"])
    g.add_argument("--flow-scale", type=float, default=0.5, help="光流分辨率比例")
    g.add_argument("--flow-iters", type=int, default=8, help="RAFT 迭代次数上限")
    g.add_argument("--no-early-exit", action="store_true",
                   help="关闭收敛提前退出（默认开启，收敛即停）")
    g.add_argument("--fp16", action="store_true",
                   help="RAFT 半精度推理（约快 20%%，长位移偶尔掉精度）")
    g.add_argument("--sigma", type=float, default=1.5, help="前后向一致性容差（像素）")
    g.add_argument("--flow-dirs", default="single", choices=["single", "both"],
                   help="single=每帧算 1 次光流（快 2 倍）；both=双向一致性（更稳）")
    g.add_argument("--engine", default="flow", choices=["flow", "linear"],
                   help="flow=本方案；linear=RSMB 式线性模糊基线")

    g = p.add_argument_group("性能与输出")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    g.add_argument("--fast", action="store_true",
                   help="快速档：raft_small + 0.4 缩放 + 8 采样")
    g.add_argument("--preview", type=int, default=0, metavar="H",
                   help="按指定高度快速预览（例如 720）")
    g.add_argument("--codec", default="auto",
                   choices=["auto", "h264_nvenc", "hevc_nvenc", "libx264", "libx265"])
    g.add_argument("--crf", type=int, default=16, help="质量因子，越小越好")
    g.add_argument("--preset", default="", help="编码器预设")
    g.add_argument("--info", action="store_true", help="只打印视频信息")
    g.add_argument("--trim", nargs=2, type=float, metavar=("START", "END"),
                   help="只处理该时间区间（秒），如 --trim 3.5 12.0")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"错误：找不到文件 {args.input}", file=sys.stderr)
        return 2

    if args.info:
        info = probe(args.input)
        print(f"文件      {info.path}")
        print(f"分辨率    {info.width}x{info.height}")
        print(f"帧率      {float(info.fps):.4f} fps  ({info.fps})")
        print(f"帧数      {info.nb_frames}")
        print(f"时长      {info.duration:.2f}s")
        print(f"编码      {info.codec} / {info.pix_fmt}")
        print(f"音轨      {'有' if info.has_audio else '无'}")
        print(f"ffmpeg    {find_exe('ffmpeg')}")
        print(f"ffprobe   {find_exe('ffprobe')}")
        return 0

    cfg = BlurConfig(
        angle=args.angle, samples=args.samples, strength=args.strength,
        occlusion=args.occlusion, photometric=args.photometric,
        jitter=not args.no_jitter,
        linear_light=not args.gamma_space,
        flow_model=args.flow_model, flow_scale=args.flow_scale,
        flow_iters=args.flow_iters, flow_dirs=args.flow_dirs,
        cons_sigma=args.sigma, early_exit=not args.no_early_exit,
        flow_fp16=args.fp16, bicubic=args.bicubic,
        device=args.device, codec=args.codec, crf=args.crf, preset=args.preset,
        preview_height=args.preview,
    )
    if args.trim:
        cfg.trim_start, cfg.trim_end = args.trim
    if args.fast:
        cfg.flow_model = "raft_small"
        cfg.flow_scale = 0.4
        cfg.flow_iters = 6
        cfg.samples = min(cfg.samples, 8)
    if args.robust:
        cfg.flow_dirs = "both"
        cfg.occlusion = max(cfg.occlusion, 0.4)
        cfg.photometric = max(cfg.photometric, 0.4)

    out = args.output or _default_output(args.input, args.preview or args.fast)

    renderer = MotionBlurRenderer(cfg)
    t0 = time.time()
    last = [0.0]

    def progress(d):
        now = time.time()
        if now - last[0] < 0.3 and d["done"] != d["total"]:
            return
        last[0] = now
        pct = 100.0 * d["done"] / d["total"] if d["total"] else 0.0
        bar_n = int(pct / 2.5)
        bar = "#" * bar_n + "-" * (40 - bar_n)
        sys.stdout.write(f"\r[{bar}] {pct:5.1f}%  {d['fps']:5.1f} fps  "
                         f"剩余 {_fmt(d['eta'])}   ")
        sys.stdout.flush()

    try:
        stats = renderer.render(args.input, out, progress=progress,
                                log=lambda s: print(f"  {s}"))
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\n渲染失败：{exc}", file=sys.stderr)
        return 1
    finally:
        renderer.close()

    print(f"\r{' ' * 80}\r", end="")
    if stats.get("cancelled"):
        return 130
    print(f"输出 {stats['output']}")
    print(f"{stats['frames']} 帧 / {stats['seconds']:.1f}s = {stats['fps']:.2f} fps")
    prof = stats.get("profile") or {}
    if prof.get("frames_profiled"):
        print(f"分段均值（前 {prof['frames_profiled']} 帧）："
              f"光流 {prof.get('flow_ms', 0):.0f} ms · "
              f"积分 {prof.get('render_ms', 0):.0f} ms · "
              f"解码 {prof.get('dec_ms', 0):.0f} ms · "
              f"编码 {prof.get('enc_ms', 0):.0f} ms")
    return 0


def _fmt(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:d}:{sec % 60:02d}"


def _default_output(src: str, quick: bool) -> str:
    stem, _ = os.path.splitext(src)
    return stem + ("_preview.mp4" if quick else "_blur.mp4")


if __name__ == "__main__":
    raise SystemExit(main())

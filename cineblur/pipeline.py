"""渲染流水线：解码 -> 光流 -> 快门积分 -> 编码，四段并行。

性能相关的几处设计
------------------
1. **线性光帧缓存**。同一帧在一次渲染中会作为 prev / cur / next 被用三次，
   若每次都在积分函数里做 sRGB 解码，等于把色彩转换做了三遍。这里按帧号缓存，
   每帧只转换一次，3 张图的开销压到 1 张。
2. **光流特征复用**。相邻两对光流的中间帧特征只编码一次（见 flow.py）。
3. **分段计时**。用 CUDA event 分别统计光流与积分，结果随 stats 返回，
   界面可以直接显示瓶颈在哪一段，而不是一个笼统的 fps。
"""
from __future__ import annotations

import os
import threading
import time
from queue import Empty, Queue
from typing import Callable

import numpy as np
import torch

from .blur import render_frame
from .config import BlurConfig
from .flow import FlowEngine
from .imageops import srgb_to_linear
from .media import Decoder, Encoder, VideoInfo, probe

PROFILE_FRAMES = 48          # 只统计前若干帧，避免长时间渲染被计时器拖慢


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 -> [1,3,H,W] float(0..1)。单次 H2D，转换留在 GPU 上做。"""
    t = torch.from_numpy(arr).to(device, non_blocking=True)
    return t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


@torch.inference_mode()
def _to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.clamp_(0, 1).mul_(255.0).round_().to(torch.uint8)
    return t[0].permute(1, 2, 0).contiguous().cpu().numpy()


def target_size(info: VideoInfo, preview_height: int = 0) -> tuple[int, int]:
    w, h = info.width, info.height
    if preview_height and h > preview_height:
        h = preview_height
        w = int(round(info.width * preview_height / info.height))
    w -= w % 2
    h -= h % 2
    return max(2, w), max(2, h)


class MotionBlurRenderer:
    def __init__(self, cfg: BlurConfig):
        cfg.validate()
        self.cfg = cfg
        self.engine: FlowEngine | None = None

    # ------------------------------------------------------------ 引擎
    def _ensure_engine(self):
        if self.engine is None:
            self.engine = FlowEngine(model=self.cfg.flow_model, iters=self.cfg.flow_iters,
                                     device=self.cfg.device, scale=self.cfg.flow_scale,
                                     early_exit=self.cfg.early_exit,
                                     fp16=self.cfg.flow_fp16)
        return self.engine

    # ------------------------------------------------------------ 主流程
    def render(self, src: str, dst: str,
               progress: Callable[[dict], None] | None = None,
               cancel: Callable[[], bool] | None = None,
               log: Callable[[str], None] | None = None) -> dict:
        cfg = self.cfg
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        log = log or (lambda s: None)
        cancel_started = time.time()
        if cancel and cancel():
            log("已取消")
            return {"cancelled": True, "frames": 0,
                    "seconds": time.time() - cancel_started, "profile": {}}
        info = probe(src)
        w, h = target_size(info, cfg.preview_height)
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)

        fps_f = float(info.fps)
        ss = max(0.0, cfg.trim_start)
        end = cfg.trim_end if cfg.trim_end > ss else 0.0
        dur = (end - ss) if end else None

        bypass_flow = cfg.shutter_fraction() * max(0.0, cfg.strength) <= 1e-6
        engine = None if bypass_flow else self._ensure_engine()
        if engine is not None:
            engine.drop_cache()
            # 在计时开始前完成权重加载：否则短片段会把 2~3 秒的模型加载摊进 fps，
            # 报出一个远低于真实的数字。
            _ = engine.net
            if cancel and cancel():
                log("已取消")
                return {"cancelled": True, "frames": 0,
                        "seconds": time.time() - cancel_started, "profile": {}}
        log(f"输入 {info.width}x{info.height} @ {fps_f:.3f}fps  "
            f"{info.nb_frames} 帧  处理尺寸 {w}x{h}")
        if ss > 0 or dur:
            log(f"裁剪 {ss:.2f}s → {end if end else info.duration:.2f}s")
        if engine is not None:
            log(f"光流 {cfg.flow_model} @ {cfg.flow_scale:g}x  迭代 {cfg.flow_iters}"
                f"{'（收敛早退）' if cfg.early_exit else ''}  设备 {engine.device}")
        else:
            log("快门角为 0°，跳过光流")
        log(f"快门角 {cfg.angle:g}° ({cfg.shutter_fraction():.3f} 帧)  "
            f"采样 {cfg.samples}  抖动 {'开' if cfg.jitter else '关'}")

        dec = Decoder(src, w, h, ss=ss, dur=dur)
        enc = Encoder(dst, w, h, info.fps, codec=cfg.codec, crf=cfg.crf,
                      preset=cfg.preset, audio_from=src if info.has_audio else None,
                      audio_ss=ss, audio_dur=dur,
                      color={"color_space": info.color_space,
                             "color_primaries": info.color_primaries,
                             "color_transfer": info.color_transfer})

        t0 = time.time()
        if dur:
            total = max(1, int(round(dur * fps_f)))
        else:
            total = max(1, (info.nb_frames or 0) - int(round(ss * fps_f)))

        if bypass_flow:
            done = 0
            cancelled = False
            try:
                while True:
                    if cancel and cancel():
                        cancelled = True
                        break
                    frame = dec.read()
                    if frame is None:
                        break
                    enc.write(frame)
                    done += 1
                    _report(progress, done, total, t0)
            finally:
                dec.close()
            ok, err = enc.close()
            elapsed = time.time() - t0
            if cancelled:
                log("已取消")
                return {"cancelled": True, "frames": done, "seconds": elapsed,
                        "profile": {}}
            if not ok:
                raise RuntimeError(f"编码失败：{err.strip()[-800:]}")
            stats = {
                "frames": done, "seconds": elapsed,
                "fps": done / max(1e-6, elapsed),
                "output": dst, "size": f"{w}x{h}", "profile": {},
            }
            log(f"完成：{done} 帧 / {elapsed:.1f}s = {stats['fps']:.2f} fps -> {dst}")
            return stats

        # ---- 读线程 ----
        q_raw: Queue = Queue(maxsize=6)
        stop = threading.Event()
        stat = {"dec": 0.0, "dec_n": 0, "enc": 0.0, "enc_n": 0}

        def reader():
            try:
                while not stop.is_set():
                    t0 = time.perf_counter()
                    f = dec.read()
                    stat["dec"] += time.perf_counter() - t0
                    if f is None:
                        break
                    stat["dec_n"] += 1
                    q_raw.put(f)
            finally:
                q_raw.put(None)

        # ---- 写线程 ----
        q_out: Queue = Queue(maxsize=6)

        def writer():
            while True:
                item = q_out.get()
                if item is None:
                    break
                t0 = time.perf_counter()
                enc.write(item)
                stat["enc"] += time.perf_counter() - t0
                stat["enc_n"] += 1

        tr = threading.Thread(target=reader, daemon=True)
        tw = threading.Thread(target=writer, daemon=True)

        device = engine.device

        tr.start()
        tw.start()
        done = 0
        cancelled = False

        # ---- 线性光帧缓存 ----
        lin: dict[int, torch.Tensor] = {}
        use_linear = cfg.linear_light

        def lin_of(idx: int, t: torch.Tensor) -> torch.Tensor:
            v = lin.get(idx)
            if v is None:
                v = srgb_to_linear(t) if use_linear else t
                lin[idx] = v
                while len(lin) > 3:
                    lin.pop(next(iter(lin)))
            return v

        # ---- 分段计时 ----
        ev = {"flow": [], "render": []}
        do_profile = device.type == "cuda"

        def mark(key):
            # 前 3 帧含权重加载与 cuDNN 算子调优，计入均值会严重高估
            if not do_profile or done < 3 or done > PROFILE_FRAMES:
                return None
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        try:
            first = _wait(q_raw)
            if first is None:
                raise RuntimeError("未能解码出任何帧")
            second = _wait(q_raw)
            both = cfg.flow_dirs == "both"

            idx_cur, idx_next = 0, 1
            cur_t = _to_tensor(first, device)
            next_t = _to_tensor(second, device) if second is not None else cur_t
            prev_t = cur_t
            f_cur_prev = f_prev_cur = f_next_cur = None

            e0 = mark("flow")
            f_cur_next = engine.flow(cur_t, next_t, idx_cur, idx_next)
            if both:
                f_next_cur = engine.flow(next_t, cur_t, idx_next, idx_cur)
            e1 = mark("flow")
            if e0 is not None:
                ev["flow"].append((e0, e1))

            while True:
                if cancel and cancel():
                    cancelled = True
                    break

                # 预读一帧：只用于下一次前移，不参与本次渲染
                nxt_arr = _wait(q_raw)
                eof = nxt_arr is None

                e0 = mark("render")
                cur_l = lin_of(idx_cur, cur_t)
                prev_l = cur_l if idx_cur == 0 else lin_of(idx_cur - 1, prev_t)
                out = render_frame(prev_t, cur_t, next_t, f_cur_next, f_cur_prev,
                                   cfg, f_nb=f_next_cur, f_pf=f_prev_cur,
                                   linear=(prev_l, cur_l, lin_of(idx_next, next_t)))
                e1 = mark("render")
                if e0 is not None:
                    ev["render"].append((e0, e1))

                q_out.put(_to_numpy(out))
                done += 1
                _report(progress, done, total, t0)

                if eof:
                    # 尾帧：无后继帧，沿用上一对光流，保证输出帧数与输入一致
                    out = render_frame(cur_t, next_t, next_t, f_cur_next, None, cfg,
                                       linear=(lin_of(idx_cur, cur_t),
                                               lin_of(idx_next, next_t),
                                               lin_of(idx_next, next_t)))
                    q_out.put(_to_numpy(out))
                    done += 1
                    _report(progress, done, total, t0)
                    break

                idx_prev, idx_cur = idx_cur, idx_next
                idx_next += 1
                prev_t, cur_t = cur_t, next_t
                next_t = _to_tensor(nxt_arr, device)
                if both:
                    f_prev_cur, f_cur_prev = f_cur_next, f_next_cur

                e0 = mark("flow")
                f_cur_next = engine.flow(cur_t, next_t, idx_cur, idx_next)
                if both:
                    f_next_cur = engine.flow(next_t, cur_t, idx_next, idx_cur)
                e1 = mark("flow")
                if e0 is not None:
                    ev["flow"].append((e0, e1))
        finally:
            stop.set()
            q_out.put(None)
            tw.join(timeout=1800)
            tr.join(timeout=30)
            dec.close()

        ok, err = enc.close()
        elapsed = time.time() - t0
        profile = {}
        if do_profile:
            try:
                torch.cuda.synchronize()
                for k, pairs in ev.items():
                    if pairs:
                        profile[k + "_ms"] = sum(a.elapsed_time(b) for a, b in pairs) / len(pairs)
            except Exception:  # noqa: BLE001
                pass
        for k, nk in (("dec", "dec_n"), ("enc", "enc_n")):
            if stat[nk]:
                profile[k + "_ms"] = stat[k] / stat[nk] * 1000.0
        if profile:
            profile["frames_profiled"] = min(done, PROFILE_FRAMES)

        if cancelled:
            log("已取消")
            return {"cancelled": True, "frames": done, "seconds": elapsed,
                    "profile": profile}
        if not ok:
            raise RuntimeError(f"编码失败：{err.strip()[-800:]}")
        stats = {
            "frames": done, "seconds": elapsed,
            "fps": done / max(1e-6, elapsed),
            "output": dst, "size": f"{w}x{h}",
            "profile": profile,
        }
        log(f"完成：{done} 帧 / {elapsed:.1f}s = {stats['fps']:.2f} fps -> {dst}")
        if profile.get("flow_ms"):
            log(f"分段：光流 {profile.get('flow_ms', 0):.0f} ms · "
                f"积分 {profile.get('render_ms', 0):.0f} ms · "
                f"解码 {profile.get('dec_ms', 0):.0f} ms · "
                f"编码 {profile.get('enc_ms', 0):.0f} ms")
        return stats

    def close(self):
        if self.engine:
            self.engine.release()
            self.engine = None


def _wait(q: Queue):
    while True:
        try:
            return q.get(timeout=0.2)
        except Empty:
            continue


def _report(progress, done: int, total: int, t0: float):
    if not progress:
        return
    elapsed = max(1e-6, time.time() - t0)
    progress({
        "done": done, "total": total,
        "fps": done / elapsed,
        "elapsed": elapsed,
        "eta": (total - done) / (done / elapsed) if total and done else 0.0,
    })

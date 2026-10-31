"""Render depth map videos from decoded RGB frames."""
from __future__ import annotations

import os
import threading
import time
from queue import Empty, Full, Queue
from typing import Callable

import numpy as np
import torch

from .config import DepthConfig
from .depth_engine import DepthEngine
from .depth_ops import depth_range, normalize_depth, smooth_range
from .media import Decoder, Encoder, VideoInfo, probe


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert an HWC uint8 frame to a normalized NCHW RGB tensor."""
    tensor = torch.from_numpy(arr).to(device, non_blocking=True)
    return tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def _target_size(info: VideoInfo, output_height: int) -> tuple[int, int]:
    width, height = info.width, info.height
    if output_height and height > output_height:
        height = output_height
        width = int(round(info.width * output_height / info.height))
    width -= width % 2
    height -= height % 2
    return max(2, width), max(2, height)


def _put(queue: Queue, item, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            queue.put(item, timeout=0.2)
            return True
        except Full:
            continue
    return False


def _wait(queue: Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            return queue.get(timeout=0.2)
        except Empty:
            continue
    return None


def _report(progress, done: int, total: int, started: float) -> None:
    if not progress:
        return
    elapsed = max(1e-6, time.time() - started)
    rate = done / elapsed
    progress(
        {
            "done": done,
            "total": total,
            "fps": rate,
            "elapsed": elapsed,
            "eta": (total - done) / rate if total and rate else 0.0,
        }
    )


class DepthRenderer:
    def __init__(self, cfg: DepthConfig, engine: DepthEngine | None = None):
        cfg.validate()
        self.cfg = cfg
        self.engine = engine

    def _ensure_engine(self):
        if self.engine is None:
            self.engine = DepthEngine(device=self.cfg.device)
        return self.engine

    @torch.inference_mode()
    def render(
        self,
        src: str,
        dst: str,
        progress: Callable[[dict], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> dict:
        cfg = self.cfg
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        info = probe(src)
        width, height = _target_size(info, cfg.output_height)
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)

        trim_start = max(0.0, cfg.trim_start)
        trim_end = cfg.trim_end if cfg.trim_end > trim_start else 0.0
        duration = (trim_end - trim_start) if trim_end else None
        fps = info.fps
        fps_value = float(fps)
        engine = self._ensure_engine()
        log = log or (lambda message: None)

        log(
            f"Input {info.width}x{info.height} @ {fps_value:.3f}fps; "
            f"processing {width}x{height}"
        )
        if trim_start or duration:
            end = trim_end if trim_end else info.duration
            log(f"Trim {trim_start:.2f}s -> {end:.2f}s")
        log(f"Depth quality {cfg.quality}; device {engine.device}")

        decoder = Decoder(src, width, height, ss=trim_start, dur=duration)
        try:
            encoder = Encoder(
                dst,
                width,
                height,
                fps,
                codec=cfg.codec,
                crf=cfg.crf,
                audio_from=None,
            )
        except BaseException:
            decoder.close()
            raise

        if duration:
            total = max(1, int(round(duration * fps_value)))
        else:
            total = max(1, (info.nb_frames or 0) - int(round(trim_start * fps_value)))

        raw_queue: Queue = Queue(maxsize=6)
        output_queue: Queue = Queue(maxsize=6)
        stop = threading.Event()
        reader_errors: list[BaseException] = []
        writer_errors: list[BaseException] = []
        timing = {"dec": 0.0, "dec_n": 0, "enc": 0.0, "enc_n": 0}

        def reader() -> None:
            try:
                while not stop.is_set():
                    started = time.perf_counter()
                    frame = decoder.read()
                    timing["dec"] += time.perf_counter() - started
                    if frame is None:
                        break
                    timing["dec_n"] += 1
                    if not _put(raw_queue, frame, stop):
                        break
            except BaseException as exc:
                reader_errors.append(exc)
            finally:
                _put(raw_queue, None, stop)

        def writer() -> None:
            while True:
                try:
                    frame = output_queue.get(timeout=0.2)
                except Empty:
                    if stop.is_set():
                        break
                    continue
                if frame is None:
                    break
                try:
                    started = time.perf_counter()
                    encoder.write(frame)
                    timing["enc"] += time.perf_counter() - started
                    timing["enc_n"] += 1
                except BaseException as exc:
                    writer_errors.append(exc)
                    stop.set()
                    break

        reader_thread = threading.Thread(target=reader, daemon=True)
        writer_thread = threading.Thread(target=writer, daemon=True)
        started = time.time()
        done = 0
        cancelled = False
        previous_range = None
        threads_started = False

        try:
            reader_thread.start()
            writer_thread.start()
            threads_started = True
            while True:
                if stop.is_set():
                    break
                if cancel and cancel():
                    cancelled = True
                    break
                frame = _wait(raw_queue, stop)
                if frame is None:
                    break

                frame_tensor = _to_tensor(frame, engine.device)
                depth = engine.estimate(frame_tensor, cfg.input_size())
                current_range = depth_range(depth)
                current_range = smooth_range(previous_range, current_range)
                previous_range = current_range
                gray = normalize_depth(
                    depth,
                    current_range[0],
                    current_range[1],
                    cfg.invert,
                )
                gray_u8 = (
                    gray.clamp(0, 1)[0]
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                rgb = np.repeat(gray_u8[..., None], 3, axis=2)
                if not _put(output_queue, rgb, stop):
                    break
                done += 1
                _report(progress, done, total, started)
        finally:
            stop.set()
            if threads_started:
                try:
                    output_queue.put_nowait(None)
                except Full:
                    pass
                writer_thread.join(timeout=1800)
                reader_thread.join(timeout=30)
            decoder.close()
            ok, error = encoder.close()

        if reader_errors:
            raise RuntimeError("Depth video decoding failed") from reader_errors[0]
        if writer_errors:
            raise RuntimeError("Depth video encoding failed") from writer_errors[0]
        if not cancelled and done == 0:
            raise RuntimeError("No frames were decoded")
        if not ok:
            raise RuntimeError(f"Encoding failed: {error.strip()[-800:]}")

        elapsed = time.time() - started
        profile = {}
        for key, count_key in (("dec", "dec_n"), ("enc", "enc_n")):
            if timing[count_key]:
                profile[f"{key}_ms"] = (
                    timing[key] / timing[count_key] * 1000.0
                )
        if profile:
            profile["frames_profiled"] = done

        stats = {
            "frames": done,
            "seconds": elapsed,
            "fps": done / max(1e-6, elapsed),
            "output": dst,
            "size": f"{width}x{height}",
            "profile": profile,
        }
        if cancelled:
            stats["cancelled"] = True
            log("Cancelled")
        else:
            log(
                f"Completed: {done} frames / {elapsed:.1f}s "
                f"= {stats['fps']:.2f} fps -> {dst}"
            )
        return stats

    def close(self) -> None:
        if self.engine is not None:
            self.engine.release()
            self.engine = None

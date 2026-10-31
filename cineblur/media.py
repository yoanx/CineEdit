"""媒体 I/O：ffmpeg / ffprobe 定位、探测、解码管道、编码管道。

不依赖 OpenCV，直接走 ffmpeg 的 rawvideo 管道，零额外拷贝开销；
解码、GPU 运算、编码三者在独立线程中流水并行。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


def unique_output_path(path: str) -> str:
    """Return a numbered sibling path when the requested output already exists."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    index = 2
    while True:
        candidate = f"{stem}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


# ---------------------------------------------------------------- 可执行文件定位

def _candidate_dirs() -> list[str]:
    """按优先级返回可能包含 ffmpeg/ffprobe 的目录。

    打包分发时优先用随包的静态构建：别人的机器上通常既没有 PATH 里的 ffmpeg，
    也没有 ``~/tools``，而 imageio-ffmpeg 只带 ffmpeg、不带 ffprobe。
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs: list[str] = []

    # 1) 随包目录（onedir 打包后 assets/ffmpeg，或 exe 旁的 ffmpeg/）
    if getattr(sys, "frozen", False):
        root = os.path.dirname(os.path.abspath(sys.executable))
        dirs += [os.path.join(root, "ffmpeg"),
                 os.path.join(root, "assets", "ffmpeg"),
                 os.path.join(root, "_internal", "ffmpeg"),
                 os.path.join(root, "_internal", "assets", "ffmpeg")]
    if hasattr(sys, "_MEIPASS"):
        dirs += [os.path.join(sys._MEIPASS, "assets", "ffmpeg"),
                 os.path.join(sys._MEIPASS, "ffmpeg")]
    dirs.append(os.path.join(here, "assets", "ffmpeg"))   # 源码目录

    # 2) 开发机上的常见位置
    home = os.path.expanduser("~")
    dirs += [
        os.path.join(home, "tools", "ffmpeg", "bin"),
        os.path.join(os.getcwd(), "tools", "ffmpeg", "bin"),
        os.path.join(here, "tools", "ffmpeg", "bin"),
        r"C:\ffmpeg\bin",
    ]
    tools = os.path.join(home, "tools")
    if os.path.isdir(tools):
        for name in os.listdir(tools):
            p = os.path.join(tools, name, "bin")
            if os.path.isdir(p):
                dirs.append(p)
    return dirs


def find_exe(name: str) -> str | None:
    suffix = ".exe" if sys.platform == "win32" else ""
    # 打包分发时优先随包构建：系统里可能装着旧版 ffmpeg，而 -fps_mode 等参数
    # 在 4.x 上不存在，让它抢先会直接报错。
    if getattr(sys, "frozen", False):
        for d in _candidate_dirs():
            exe = os.path.join(d, name + suffix)
            if os.path.exists(exe):
                return exe
    p = shutil.which(name)
    if p:
        return p
    # 优先使用完整构建（同时带 ffmpeg/ffprobe），避免版本混用
    for d in _candidate_dirs():
        exe = os.path.join(d, name + suffix)
        if os.path.exists(exe):
            return exe
    if name in ("ffmpeg", "ffprobe"):
        # 同目录推导：找到其中一个，另一个通常就在旁边
        other = find_exe("ffprobe" if name == "ffmpeg" else "ffmpeg")
        if other:
            cand = os.path.join(os.path.dirname(other),
                                name + (".exe" if sys.platform == "win32" else ""))
            if os.path.exists(cand):
                return cand
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            p = imageio_ffmpeg.get_ffmpeg_exe()
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return None


CREATIONFLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ---------------------------------------------------------------- 探测

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction
    nb_frames: int
    duration: float
    has_audio: bool
    codec: str = ""
    pix_fmt: str = ""
    color_space: str = ""
    color_primaries: str = ""
    color_transfer: str = ""


def probe(path: str) -> VideoInfo:
    ffprobe = find_exe("ffprobe")
    if ffprobe:
        cmd = [ffprobe, "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", path]
        out = subprocess.run(cmd, capture_output=True, timeout=60,
                             creationflags=CREATIONFLAGS)
        if out.returncode == 0:
            data = json.loads(out.stdout.decode("utf-8", "ignore"))
            vs = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
            if not vs:
                raise RuntimeError("文件里没有视频流")
            v = vs[0]
            fps = _parse_fps(v.get("r_frame_rate") or v.get("avg_frame_rate") or "25/1")
            dur = float(data.get("format", {}).get("duration") or v.get("duration") or 0.0)
            nb = int(v.get("nb_frames") or 0)
            if nb <= 0 and dur > 0:
                nb = int(round(dur * float(fps)))
            return VideoInfo(
                path=path, width=int(v["width"]), height=int(v["height"]), fps=fps,
                nb_frames=nb, duration=dur,
                has_audio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
                codec=v.get("codec_name", ""), pix_fmt=v.get("pix_fmt", ""),
                color_space=v.get("color_space", "") or "",
                color_primaries=v.get("color_primaries", "") or "",
                color_transfer=v.get("color_transfer", "") or "",
            )
    # 兜底：解析 ffmpeg -i 的 stderr
    ffmpeg = find_exe("ffmpeg") or ""
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True,
                         timeout=60, creationflags=CREATIONFLAGS)
    text = (out.stderr or b"").decode("utf-8", "ignore")
    m = re.search(r"Stream #\d+:\d+.*?Video:.*?(\d{2,5})x(\d{2,5})", text)
    if not m:
        raise RuntimeError("无法解析视频信息")
    fps_m = re.search(r"(\d+(?:\.\d+)?)\s+fps", text)
    fps = Fraction(fps_m.group(1)).limit_denominator(1001) if fps_m else Fraction(25, 1)
    dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    dur = (int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))) \
        if dur_m else 0.0
    return VideoInfo(path=path, width=int(m.group(1)), height=int(m.group(2)), fps=fps,
                     nb_frames=int(dur * float(fps)) if dur else 0, duration=dur,
                     has_audio="Audio:" in text, codec="", pix_fmt="")


def _parse_fps(s: str) -> Fraction:
    try:
        if "/" in s:
            num, den = s.split("/")
            f = Fraction(int(num), int(den))
        else:
            f = Fraction(s).limit_denominator(1001)
        return f if f > 0 else Fraction(25, 1)
    except Exception:
        return Fraction(25, 1)


# ---------------------------------------------------------------- 解码 / 编码

class Decoder:
    """按帧读取 rgb24 原始数据。"""

    def __init__(self, path: str, width: int, height: int,
                 ss: float = 0.0, dur: float | None = None):
        self.path, self.w, self.h = path, width, height
        self.frame_bytes = width * height * 3
        ffmpeg = find_exe("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("找不到 ffmpeg，请先安装（见 README）")
        cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin"]
        if ss > 0:
            cmd += ["-ss", f"{ss:.4f}"]
        cmd += ["-i", path]
        if dur:
            cmd += ["-t", f"{dur:.4f}"]
        cmd += ["-vf", f"scale={width}:{height}:flags=lanczos,format=rgb24",
                "-fps_mode", "passthrough",
                "-f", "rawvideo", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     creationflags=CREATIONFLAGS, bufsize=self.frame_bytes * 2)

    def read(self) -> np.ndarray | None:
        buf = self.proc.stdout.read(self.frame_bytes)
        if not buf or len(buf) < self.frame_bytes:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3).copy()

    def close(self):
        for pipe in (self.proc.stdout, self.proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Encoder:
    """接收 rgb24 帧并编码；若提供源文件则复用其音轨。"""

    def __init__(self, path: str, width: int, height: int, fps: Fraction,
                 codec: str = "auto", crf: int = 16, preset: str = "",
                 audio_from: str | None = None, color: dict | None = None,
                 audio_ss: float = 0.0, audio_dur: float | None = None):
        ffmpeg = find_exe("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("找不到 ffmpeg，请先安装（见 README）")
        self.w, self.h = width, height
        self.frame_bytes = width * height * 3

        vcodec, vopts = _resolve_codec(ffmpeg, codec, crf, preset)
        cmd = [ffmpeg, "-hide_banner", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}",
               "-i", "-"]
        if audio_from:
            if audio_ss > 0:
                cmd += ["-ss", f"{audio_ss:.4f}"]
            cmd += ["-i", audio_from]
            if audio_dur:
                cmd += ["-t", f"{audio_dur:.4f}"]
        cmd += ["-map", "0:v:0"]
        if audio_from:
            cmd += ["-map", "1:a:0?", "-c:a", "copy"]
        cmd += ["-c:v", vcodec] + vopts + ["-pix_fmt", "yuv420p"]
        color = color or {}
        for key, flag in (("color_space", "-colorspace"),
                          ("color_primaries", "-color_primaries"),
                          ("color_transfer", "-color_trc")):
            val = color.get(key)
            if val and val != "unknown":
                cmd += [flag, val]
        cmd += ["-movflags", "+faststart", path]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     creationflags=CREATIONFLAGS,
                                     bufsize=self.frame_bytes)

    def write(self, frame_u8: np.ndarray):
        self.proc.stdin.write(frame_u8.tobytes())

    def close(self) -> tuple[bool, str]:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        err = b""
        try:
            _, err = self.proc.communicate(timeout=1800)
        except Exception:
            self.proc.kill()
        ok = self.proc.returncode == 0
        return ok, (err or b"").decode("utf-8", "ignore")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.proc.poll() is None:
            self.close()


def _encoders(ffmpeg: str) -> str:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, timeout=30,
                             creationflags=CREATIONFLAGS)
        return (out.stdout or b"").decode("utf-8", "ignore")
    except Exception:
        return ""


def _resolve_codec(ffmpeg: str, codec: str, crf: int, preset: str):
    avail = _encoders(ffmpeg)

    def has(name: str) -> bool:
        return name in avail

    if codec == "auto":
        codec = "h264_nvenc" if has("h264_nvenc") else "libx264"

    if codec == "h264_nvenc" and has("h264_nvenc"):
        return codec, ["-preset", preset or "p6", "-tune", "hq",
                       "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    if codec == "hevc_nvenc" and has("hevc_nvenc"):
        return codec, ["-preset", preset or "p6", "-tune", "hq",
                       "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    if codec == "libx265" and has("libx265"):
        return codec, ["-crf", str(crf), "-preset", preset or "medium"]
    return "libx264", ["-crf", str(crf), "-preset", preset or "slow"]

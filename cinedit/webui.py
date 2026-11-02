"""CineEdit 控制台（本地 HTTP + 单页界面）。

    python -m cinedit.app            # 独立窗口（需要 pywebview，缺失时回退浏览器）
    python -m cinedit.webui          # 仅启动 HTTP 服务

只监听 127.0.0.1，不对外暴露。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import string
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from .config import BlurConfig, DepthConfig
from .depth_pipeline import DepthRenderer
from .media import find_exe, probe, unique_output_path
from .pipeline import MotionBlurRenderer
from .weights import video_depth_model_available

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}

JOB: dict = {
    "running": False, "progress": {"done": 0, "total": 0, "fps": 0.0, "eta": 0.0},
    "log": [], "result": None, "error": None, "output": "",
}
_LOCK = threading.Lock()
_CANCEL = threading.Event()
_PREVIEW_LOCK = threading.Lock()


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1],
                      (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


# ---------------------------------------------------------------- 配置

# 档位由 tools/tradeoff.py 的实测矩阵决定：精度在 0.5 倍处基本饱和，
# 再往上加分辨率只增加耗时；迭代数在 12 附近收益明显，超过 16 趋于平台。
QUALITY = {
    "turbo":  {"flow_model": "raft_small", "flow_scale": 0.4, "flow_iters": 4},
    "fast":   {"flow_model": "raft_small", "flow_scale": 0.5, "flow_iters": 6},
    "std":    {"flow_model": "raft_large", "flow_scale": 0.5, "flow_iters": 8},
    "fine":   {"flow_model": "raft_large", "flow_scale": 0.7, "flow_iters": 12},
}


def _num(v, d: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _flag(p: dict, key: str, default: bool = False) -> bool:
    v = p.get(key)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "on", "yes")


def save_upload(stream, filename: str, length: int, dest_dir: str | None = None) -> str:
    name = os.path.basename(filename or "video.mp4")
    if os.path.splitext(name)[1].lower() not in VIDEO_EXT:
        raise ValueError("unsupported video extension")
    root = dest_dir or os.path.join(tempfile.gettempdir(), "cinedit_uploads")
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, f"{uuid.uuid4().hex}_{name}")
    remaining = max(0, int(length))
    try:
        with open(dst, "wb") as f:
            if remaining:
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            else:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        if os.path.getsize(dst) == 0:
            raise ValueError("empty upload")
        return dst
    except Exception:
        try:
            os.remove(dst)
        except OSError:
            pass
        raise


def build_cfg(p: dict, preview: bool = False) -> BlurConfig:
    q = QUALITY.get(p.get("quality", "std"), QUALITY["std"]).copy()
    robust = _flag(p, "robust")
    cfg = BlurConfig(
        angle=_num(p.get("angle"), 180),
        samples=int(_num(p.get("samples"), 16)),
        strength=_num(p.get("strength"), 1.0),
        photometric=_num(p.get("photometric"), 0.0),
        jitter=not _flag(p, "no_jitter"),
        bicubic=_flag(p, "bicubic"),
        flow_dirs="both" if robust else "single",
        flow_model=q["flow_model"],
        flow_scale=q["flow_scale"],
        flow_iters=q["flow_iters"],
        early_exit=not _flag(p, "no_early_exit"),
        flow_fp16=_flag(p, "fp16"),
        device=p.get("device") or "auto",
        codec=p.get("codec") or "auto",
        preview_height=int(_num(p.get("preview"), 0)),
        trim_start=_num(p.get("trim_start"), 0.0),
        trim_end=_num(p.get("trim_end"), 0.0),
    )
    if robust:
        cfg.occlusion = 0.4
        cfg.photometric = max(cfg.photometric, 0.4)
    if preview:
        cfg.preview_height = 0
    return cfg


def build_depth_cfg(p: dict) -> DepthConfig:
    return DepthConfig(
        quality=p.get("quality") or "fast",
        output_height=int(_num(p.get("preview"), 0)),
        invert=_flag(p, "invert"),
        codec=p.get("codec") or "auto",
        crf=int(_num(p.get("crf"), 18)),
        device=p.get("device") or "auto",
        trim_start=_num(p.get("trim_start"), 0.0),
        trim_end=_num(p.get("trim_end"), 0.0),
    )


def default_output(src: str, mode: str) -> str:
    if mode == "depth":
        return DepthConfig().output_name(src)
    stem, _ = os.path.splitext(src)
    return stem + "_blur.mp4"


# ---------------------------------------------------------------- 环境

def env_info() -> dict:
    info = {"ffmpeg": find_exe("ffmpeg"), "gpu": None, "torch": None, "cuda": False,
            "webview": False, "depth_model": video_depth_model_available()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import webview  # noqa: F401
        info["webview"] = True
    except Exception:
        pass
    return info


def list_dir(path: str) -> dict:
    if not path:
        drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        return {"path": "", "parent": None, "dirs": drives, "files": []}
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    dirs, files = [], []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir():
                        if not e.name.startswith("$"):
                            dirs.append(e.name)
                    elif os.path.splitext(e.name)[1].lower() in VIDEO_EXT:
                        files.append(e.name)
                except OSError:
                    continue
    except PermissionError:
        pass
    dirs.sort(key=str.lower)
    files.sort(key=str.lower)
    parent = os.path.dirname(path.rstrip("\\/"))
    if not parent or parent == path:
        parent = ""
    return {"path": path, "parent": parent, "dirs": dirs, "files": files}


# ---------------------------------------------------------------- 任务

def start_job(payload: dict) -> dict:
    with _LOCK:
        if JOB["running"]:
            return {"ok": False, "error": "已有任务在运行"}
        src = (payload.get("src") or "").strip()
        dst = (payload.get("dst") or "").strip()
        mode = payload.get("mode", "blur")
        if not src or not os.path.exists(src):
            return {"ok": False, "error": "输入文件不存在"}
        if not dst:
            dst = default_output(src, mode)
        if _flag(payload, "preserve_outputs", True):
            dst = unique_output_path(dst)
        cfg = build_depth_cfg(payload) if mode == "depth" else build_cfg(payload)
        JOB.update(running=True, log=[], result=None, error=None, output=dst,
                   progress={"done": 0, "total": 0, "fps": 0.0, "eta": 0.0})
        _CANCEL.clear()

    def work():
        r = None
        try:
            r = DepthRenderer(cfg) if mode == "depth" else MotionBlurRenderer(cfg)
            stats = r.render(src, dst, progress=_set_progress,
                             cancel=_CANCEL.is_set, log=_log)
            with _LOCK:
                JOB["result"] = stats
        except Exception as exc:  # noqa: BLE001
            with _LOCK:
                JOB["error"] = str(exc)
                JOB["log"].append("错误：" + str(exc))
        finally:
            if r is not None:
                try:
                    r.close()
                except Exception as exc:  # noqa: BLE001
                    with _LOCK:
                        if not JOB["error"]:
                            JOB["error"] = str(exc)
                        JOB["log"].append("清理错误：" + str(exc))
            with _LOCK:
                JOB["running"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "output": dst}


def _set_progress(d):
    with _LOCK:
        JOB["progress"] = d


def _log(s):
    with _LOCK:
        JOB["log"].append(s)
        if len(JOB["log"]) > 400:
            JOB["log"] = JOB["log"][-400:]


def snapshot(since: int = 0) -> dict:
    with _LOCK:
        return {"running": JOB["running"], "progress": JOB["progress"],
                "log": JOB["log"][since:], "log_total": len(JOB["log"]),
                "result": JOB["result"], "error": JOB["error"],
                "output": JOB["output"]}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    # ---- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            self._send(200, ui_html().encode("utf-8"), "text/html; charset=utf-8")
        elif u.path == "/api/env":
            self._json(env_info())
        elif u.path == "/api/list":
            self._json(list_dir(q.get("path", [""])[0]))
        elif u.path == "/api/info":
            self._info(q.get("path", [""])[0])
        elif u.path == "/api/frame":
            self._frame(q)
        elif u.path == "/api/events":
            self._events(int(q.get("since", ["0"])[0]))
        else:
            self._send(404, b"not found", "text/plain")

    def _info(self, path: str):
        try:
            i = probe(path)
            self._json({"ok": True, "width": i.width, "height": i.height,
                        "fps": float(i.fps), "frames": i.nb_frames,
                        "duration": i.duration, "audio": i.has_audio,
                        "codec": i.codec})
        except Exception as exc:  # noqa: BLE001
            self._json({"ok": False, "error": str(exc)})

    def _frame(self, q):
        def g(k, d=None):
            return q.get(k, [d])[0]

        src = g("src", "")
        if not src or not os.path.exists(src):
            self._send(400, b"no src", "text/plain")
            return
        if JOB["running"]:
            self._send(409, b"busy", "text/plain")
            return
        try:
            import numpy as np  # noqa: F401
            from PIL import Image
            t = float(g("t", "0"))
            h = int(g("h", "540"))
            mode = g("mode", "blur")
            with _PREVIEW_LOCK:
                if g("orig", "0") == "1":
                    from .preview import render_original
                    arr = render_original(src, t, max_height=h)
                elif mode == "depth":
                    from .preview import render_depth_preview
                    cfg = build_depth_cfg({
                        "quality": g("quality"),
                        "invert": g("invert"),
                        "preview": g("preview"),
                        "codec": g("codec"),
                        "device": g("device"),
                    })
                    arr = render_depth_preview(src, t, cfg, max_height=h)
                else:
                    from .preview import render_preview
                    cfg = build_cfg({"angle": g("angle"), "samples": g("samples"),
                                     "strength": g("strength"),
                                     "photometric": g("photometric"),
                                     "quality": g("quality"),
                                     "bicubic": g("bicubic", "0"),
                                     "robust": g("robust")}, preview=True)
                    arr = render_preview(src, t, cfg, max_height=h)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "JPEG", quality=90)
            self._send(200, buf.getvalue(), "image/jpeg")
        except Exception as exc:  # noqa: BLE001
            self._send(500, str(exc).encode(), "text/plain")

    # ---- POST ----
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        if u.path == "/api/upload":
            try:
                path = save_upload(
                    self.rfile,
                    unquote(self.headers.get("X-Filename") or "video.mp4"),
                    n,
                )
                self._json({"ok": True, "path": path})
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            except OSError as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
            return
        payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if u.path == "/api/start":
            self._json(start_job(payload))
        elif u.path == "/api/cancel":
            _CANCEL.set()
            self._json({"ok": True})
        elif u.path == "/api/open":
            p = payload.get("path", "")
            if p and os.path.exists(p):
                if sys.platform == "win32":
                    subprocess.Popen(["explorer", "/select,", os.path.abspath(p)])
                else:
                    subprocess.Popen(["xdg-open", os.path.dirname(os.path.abspath(p))])
                self._json({"ok": True})
            else:
                self._json({"ok": False, "error": "文件不存在"})
        else:
            self._send(404, b"not found", "text/plain")

    def _events(self, since: int):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last = since
        try:
            for _ in range(36000):
                snap = snapshot(last)
                last = snap["log_total"]
                snap["since"] = last
                self.wfile.write(f"data: {json.dumps(snap, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


UI_CACHE: dict = {"mtime": 0.0, "html": ""}


def _find_ui():
    """按优先级查找 ui.html：exe 同目录（便于改样式免重打包）→ 打包内置 → 源码目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(os.path.dirname(os.path.abspath(sys.executable)))
    if hasattr(sys, "_MEIPASS"):
        roots.append(os.path.join(sys._MEIPASS, "cinedit"))
    roots += [here, os.getcwd()]
    for r in roots:
        for rel in ("ui.html", os.path.join("assets", "ui.html")):
            cand = os.path.join(r, rel)
            if os.path.isfile(cand):
                return cand
    return None


def ui_html() -> str:
    p = _find_ui()
    if not p:
        return ("<meta charset=utf-8><body style='font:13px system-ui;padding:40px'>"
                "<h1>CineEdit</h1><p>找不到 ui.html，请确认它与程序在同一目录。</p>")
    m = os.path.getmtime(p)
    if UI_CACHE["mtime"] != m:
        with io.open(p, encoding="utf-8") as f:
            UI_CACHE["html"] = f.read()
        UI_CACHE["mtime"] = m
    return UI_CACHE["html"]



def _warmup():
    try:
        from .preview import warmup
        warmup(build_cfg({"quality": "std"}))
    except Exception:  # noqa: BLE001
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="CineEdit 控制台")
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    srv = QuietThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"CineEdit 控制台：{url}")
    # 后台预载光流权重：第一次预览不必再等 5 秒
    threading.Thread(target=_warmup, daemon=True).start()
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()

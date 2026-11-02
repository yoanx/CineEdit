"""桌面应用入口：内嵌 HTTP 服务 + 独立窗口。

    python -m cinedit.app

优先用 pywebview 打开原生窗口（Windows 上走 Edge WebView2），
不可用时回退到默认浏览器，功能完全一致。
"""
from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from urllib.parse import quote

from .webui import Handler, QuietThreadingHTTPServer


class DesktopApi:
    """Methods exposed to the local UI through pywebview."""

    def select_video(self) -> str | None:
        import webview

        if not webview.windows:
            return None
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=(
                "Video files (*.mp4;*.mov;*.mkv;*.avi;*.m4v;*.webm;*.mpg;*.mpeg;*.wmv)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result


def _free_port(preferred: int = 8848) -> int:
    for p in (preferred, 0):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError:
            s.close()
    return 8848


def _log_error(text: str) -> None:
    try:
        import tempfile
        with io.open(os.path.join(tempfile.gettempdir(), "cinedit_startup.log"),
                     "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def _find_browser() -> str | None:
    """找一个支持 --app 模式的 Chromium 浏览器。"""
    cands = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def _open_app_window(url: str) -> bool:
    """用浏览器的应用模式开一个无地址栏窗口，观感等同桌面程序。"""
    exe = _find_browser()
    if not exe:
        return False
    try:
        subprocess.Popen([exe, f"--app={url}", "--window-size=1360,900",
                          "--user-data-dir=" + os.path.join(
                              os.path.expanduser("~"), ".cinedit_profile")])
        return True
    except Exception as exc:  # noqa: BLE001
        _log_error(f"app-window failed: {exc}")
        return False


def main() -> int:
    port = _free_port(8848)
    srv = QuietThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # 支持把视频拖到程序图标 / 命令行传参
    arg = next((a for a in sys.argv[1:] if os.path.isfile(a)), None)
    if arg:
        url += "?open=" + quote(os.path.abspath(arg))

    try:
        import webview
        webview.create_window(
            "CineEdit", url,
            width=1320, height=840, min_size=(1040, 660),
            background_color="#0f0f11", text_select=False,
            js_api=DesktopApi(),
        )
        webview.start()
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        _log_error("webview failed:\n" + traceback.format_exc())
        print(f"原生窗口不可用，改用应用模式窗口：{exc}")

    if _open_app_window(url):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    webbrowser.open(url)
    print(f"CineEdit 已在浏览器中打开：{url}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0

from __future__ import annotations

import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cinedit.app import DesktopApi
from cinedit.webui import save_upload


class DesktopDialogTests(unittest.TestCase):
    def test_select_video_returns_first_dialog_path(self):
        window = MagicMock()
        window.create_file_dialog.return_value = ("E:\\video\\clip.mp4",)

        with patch("webview.windows", [window]):
            path = DesktopApi().select_video()

        self.assertEqual(path, "E:\\video\\clip.mp4")
        self.assertEqual(window.create_file_dialog.call_count, 1)


class UploadTests(unittest.TestCase):
    def test_save_upload_streams_video_to_temp_directory(self):
        payload = b"video-bytes"

        with tempfile.TemporaryDirectory() as tmp:
            path = save_upload(io.BytesIO(payload), "clip.mp4", len(payload), tmp)

            self.assertEqual(os.path.dirname(path), tmp)
            self.assertTrue(os.path.basename(path).endswith("_clip.mp4"))
            with open(path, "rb") as f:
                self.assertEqual(f.read(), payload)

    def test_save_upload_rejects_non_video_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_upload(io.BytesIO(b"text"), "notes.txt", 4, tmp)


if __name__ == "__main__":
    unittest.main()

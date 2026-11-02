from __future__ import annotations

import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DepthSourceWarningUITests(unittest.TestCase):
    def test_depth_panel_contains_blurred_source_warning_and_action(self):
        with open(os.path.join(ROOT, "cinedit", "ui.html"), encoding="utf-8") as f:
            html = f.read()

        self.assertIn('id="depthSourceWarning"', html)
        self.assertIn('id="depthSourceWarningText"', html)
        self.assertIn('id="depthSourceChoose"', html)
        self.assertIn("function updateDepthSourceWarning()", html)
        self.assertIn("depthSourceWarning:", html)
        self.assertIn("chooseOriginal:", html)


if __name__ == "__main__":
    unittest.main()

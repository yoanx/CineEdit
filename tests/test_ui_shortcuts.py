from __future__ import annotations

import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UIShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "cinedit", "ui.html"), encoding="utf-8") as f:
            cls.html = f.read()

    def test_space_toggles_continuous_preview_instead_of_rendering(self):
        keydown = self.html[self.html.index('document.addEventListener("keydown"'):]
        keydown = keydown[:keydown.index("function cycleMode")]

        self.assertIn('e.key === " "', keydown)
        self.assertIn("togglePreviewPlayback()", keydown)
        self.assertNotIn("runPrimary()", keydown)

    def test_immediate_cancel_waits_for_start_acknowledgement(self):
        self.assertIn("cancelPending: false", self.html)
        self.assertIn("S.cancelPending = true", self.html)
        self.assertIn("if (S.cancelPending) requestCancel()", self.html)


if __name__ == "__main__":
    unittest.main()

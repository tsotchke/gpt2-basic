from __future__ import annotations

import contextlib
import io
import unittest

from scripts import verify_workspace_tracking
from scripts.verify_workspace_tracking import StatusEntry


class WorkspaceTrackingTests(unittest.TestCase):
    def test_self_test_probe(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            verify_workspace_tracking.self_test()

        text = output.getvalue()
        self.assertIn("PROBE_OK workspace_tracking_self_test=1", text)

    def test_parse_porcelain_z(self) -> None:
        entries = verify_workspace_tracking.parse_porcelain_z(
            b"!! qemu/staging/\0?? scratch.txt\0 M README.md\0"
        )

        self.assertEqual(
            entries,
            [
                StatusEntry("!!", "qemu/staging/"),
                StatusEntry("??", "scratch.txt"),
                StatusEntry(" M", "README.md"),
            ],
        )

    def test_ignored_category_rejects_unknown_ignored_path(self) -> None:
        self.assertIsNone(verify_workspace_tracking.ignored_category("scratch/output.log"))

    def test_ignored_category_accepts_documented_local_paths(self) -> None:
        self.assertEqual(verify_workspace_tracking.ignored_category(".DS_Store"), "os_metadata")
        self.assertEqual(verify_workspace_tracking.ignored_category(".venv-torch/"), "python_virtualenv")
        self.assertEqual(verify_workspace_tracking.ignored_category(".venv-torch/bin/python"), "python_virtualenv")
        self.assertEqual(verify_workspace_tracking.ignored_category("qemu/staging/"), "qemu_staging")
        self.assertEqual(verify_workspace_tracking.ignored_category("qemu/staging/GPT2SRC/MAIN.BAS"), "qemu_staging")


if __name__ == "__main__":
    unittest.main()

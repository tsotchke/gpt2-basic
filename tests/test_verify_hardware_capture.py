from __future__ import annotations

import contextlib
import io
from pathlib import Path
import unittest

from scripts import build_hardware_transfer
from scripts import verify_hardware_capture


ROOT = Path(__file__).resolve().parents[1]


class VerifyHardwareCaptureTests(unittest.TestCase):
    def test_verify_hardware_capture_self_test_artifacts(self) -> None:
        artifact_names = ["HWVALID.LOG", "QUAL.LOG", "PERF.LOG", "ASSIST.LOG", "ASSISTC.LOG", "HWNOTES.TXT"]
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            verify_hardware_capture.self_test()

        text = output.getvalue()
        for artifact_name in artifact_names:
            self.assertIn(artifact_name, text)
        self.assertIn("PROBE_OK hardware_capture_self_test=1", text)

    def test_hwvalid_batch_escapes_structured_log_pipes(self) -> None:
        for line in (ROOT / "hardware" / "HWVALID.BAT").read_text(encoding="ascii").splitlines():
            if line.lower().startswith("echo ") and "|" in line:
                self.assertIn("^|", line)

    def test_hardware_transfer_self_test(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            build_hardware_transfer.self_test()

        text = output.getvalue()
        self.assertIn("PROBE_OK hardware_transfer_8_3=1", text)
        self.assertIn("PROBE_OK hardware_transfer_readme=README.TXT", text)
        self.assertIn("PROBE_OK hardware_transfer_manifest=1", text)
        self.assertIn("PROBE_OK hardware_transfer_zip=1", text)
        self.assertIn("PROBE_OK hardware_transfer_zip_sha256=1", text)

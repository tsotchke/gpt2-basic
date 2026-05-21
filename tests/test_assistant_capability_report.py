from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from scripts import build_assistant_capability_report


ROOT = Path(__file__).resolve().parents[1]


class AssistantCapabilityReportTests(unittest.TestCase):
    def test_self_test(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            build_assistant_capability_report.self_test()

        self.assertIn("PROBE_OK assistant_capability_report_self_test=1", output.getvalue())

    def test_build_report_from_repository_evidence(self) -> None:
        report = build_assistant_capability_report.build_report(
            ROOT / "qemu" / "evidence",
            ROOT / "assets" / "gpt2_basic" / "PACKS",
            ROOT / "promo" / "renders" / "release-assets",
            "2026-05-21",
        )

        self.assertIn("Status: `PASS`", report)
        self.assertIn("Raw direct model prompt gate: `PASS 83/83`", report)
        self.assertIn("Hardware-capture assistant stress replies: `50`", report)
        self.assertIn("Physical machine capture status: PENDING", report)
        self.assertIn("`PORTABLE`: 11 rows", report)
        report.encode("ascii")

    def test_main_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "report.md"
            report = build_assistant_capability_report.build_report(
                ROOT / "qemu" / "evidence",
                ROOT / "assets" / "gpt2_basic" / "PACKS",
                ROOT / "promo" / "renders" / "release-assets",
                "2026-05-21",
            )
            output_path.write_text(report, encoding="ascii")

            self.assertTrue(output_path.read_text(encoding="ascii").startswith("# Assistant Capability"))


if __name__ == "__main__":
    unittest.main()

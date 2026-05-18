from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from scripts import hardware_performance_matrix
from scripts import stage_hardware_capture_evidence


class HardwarePerformanceMatrixTests(unittest.TestCase):
    def test_empty_matrix_ignores_qemu_perf_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence"
            output = root / "matrix.md"
            evidence.mkdir()
            (evidence / "perf_486_486dx2-66.log").write_text(
                "PERF_MODEL|profile=qemu|runtime_bytes=1\n"
                "PERF_SUMMARY|runs=3|tokens=3|seconds=3|tokens_per_sec=1\n",
                encoding="ascii",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                captures = hardware_performance_matrix.build_matrix(
                    evidence,
                    output,
                    require_notes=True,
                )

            self.assertEqual(captures, [])
            self.assertIn(
                "No staged physical hardware performance logs were found yet.",
                output.read_text(encoding="ascii"),
            )

    def test_matrix_reads_staged_hardware_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            evidence = root / "evidence"
            output = root / "matrix.md"
            capture.mkdir()
            stage_hardware_capture_evidence.write_sample_capture(capture)

            with contextlib.redirect_stdout(io.StringIO()):
                stage_hardware_capture_evidence.stage_capture(
                    capture,
                    evidence,
                    "486dx2_66_dos622",
                    require_assistant=True,
                    require_notes=True,
                    require_filled_notes=True,
                    force=False,
                )
                captures = hardware_performance_matrix.build_matrix(
                    evidence,
                    output,
                    require_notes=True,
                )

            self.assertEqual(len(captures), 1)
            self.assertEqual(captures[0].machine, "486dx2_66_dos622")
            self.assertEqual(captures[0].tokens_per_sec, 1.0)
            text = output.read_text(encoding="ascii")
            self.assertIn("| 486dx2_66_dos622 | 486DX2 | 66 MHz |", text)
            self.assertIn("hardware_486dx2_66_dos622_perf.log", text)

    def test_matrix_requires_paired_notes_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence"
            output = root / "matrix.md"
            evidence.mkdir()
            (evidence / "hardware_486dx2_66_dos622_perf.log").write_text(
                "PERF_MODEL|profile=486sx-safe|runtime_bytes=2055940\n"
                "PERF_SUMMARY|runs=3|tokens=127|seconds=2.69|tokens_per_sec=47.21\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit) as raised:
                hardware_performance_matrix.build_matrix(
                    evidence,
                    output,
                    require_notes=True,
                )
            self.assertIn("notes_missing=", str(raised.exception))

            with contextlib.redirect_stdout(io.StringIO()):
                captures = hardware_performance_matrix.build_matrix(
                    evidence,
                    output,
                    require_notes=False,
                )
            self.assertEqual(len(captures), 1)

    def test_self_test(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            hardware_performance_matrix.self_test()

        text = output.getvalue()
        self.assertIn("PROBE_OK hardware_performance_rows=1", text)
        self.assertIn("PROBE_OK hardware_performance_matrix_self_test=1", text)


if __name__ == "__main__":
    unittest.main()

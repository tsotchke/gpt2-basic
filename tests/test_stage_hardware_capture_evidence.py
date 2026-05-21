from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from scripts import stage_hardware_capture_evidence


class StageHardwareCaptureEvidenceTests(unittest.TestCase):
    def test_stage_capture_writes_stable_evidence_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            evidence = root / "evidence"
            capture.mkdir()
            stage_hardware_capture_evidence.write_sample_capture(capture)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                written = stage_hardware_capture_evidence.stage_capture(
                    capture,
                    evidence,
                    "486dx2_66_dos622",
                    require_assistant=True,
                    require_notes=True,
                    require_filled_notes=True,
                    force=False,
                )

            names = {path.name for path in written}
            self.assertEqual(
                names,
                {
                    "hardware_486dx2_66_dos622_capture.log",
                    "hardware_486dx2_66_dos622_quality.log",
                    "hardware_486dx2_66_dos622_perf.log",
                    "hardware_486dx2_66_dos622_assistant.log",
                    "hardware_486dx2_66_dos622_assistant_stress.log",
                    "hardware_486dx2_66_dos622_assistant_recall.log",
                    "hardware_486dx2_66_dos622_assistant_compile.log",
                    "hardware_486dx2_66_dos622_notes.md",
                    "hardware_486dx2_66_dos622_manifest.md",
                },
            )
            self.assertIn(
                "PROBE_OK hardware_capture_staged=486dx2_66_dos622",
                output.getvalue(),
            )
            notes = (evidence / "hardware_486dx2_66_dos622_notes.md").read_text(
                encoding="ascii",
            )
            manifest = (evidence / "hardware_486dx2_66_dos622_manifest.md").read_text(
                encoding="ascii",
            )
            self.assertIn("486DX2", notes)
            self.assertIn("PERF_SUMMARY|", manifest)
            self.assertIn("## File Checksums", manifest)
            self.assertIn("| SHA256 | Bytes | Path |", manifest)
            self.assertIn("hardware_486dx2_66_dos622_perf.log", manifest)
            self.assertRegex(manifest, r"`[0-9a-f]{64}`")
            stage_hardware_capture_evidence.verify_staged_manifest(evidence, "486dx2_66_dos622")

    def test_stage_capture_refuses_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            evidence = root / "evidence"
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

            with self.assertRaises(SystemExit) as raised:
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
            self.assertIn("staged_exists=", str(raised.exception))

            with contextlib.redirect_stdout(io.StringIO()):
                stage_hardware_capture_evidence.stage_capture(
                    capture,
                    evidence,
                    "486dx2_66_dos622",
                    require_assistant=True,
                    require_notes=True,
                    require_filled_notes=True,
                    force=True,
                )

    def test_verify_staged_manifest_rejects_modified_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            evidence = root / "evidence"
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

            (evidence / "hardware_486dx2_66_dos622_perf.log").write_text(
                "PERF_SUMMARY|runs=1|tokens=1|seconds=1|tokens_per_sec=1\n",
                encoding="ascii",
            )
            with self.assertRaises(SystemExit) as raised:
                stage_hardware_capture_evidence.verify_staged_manifest(evidence, "486dx2_66_dos622")
            self.assertIn("manifest_size_mismatch=hardware_486dx2_66_dos622_perf.log", str(raised.exception))

    def test_stage_capture_rejects_template_notes_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            evidence = root / "evidence"
            capture.mkdir()
            stage_hardware_capture_evidence.write_sample_capture(capture)
            (capture / "HWNOTES.TXT").write_text(
                "Machine key:\n"
                "CPU:\n"
                "Clock:\n"
                "RAM:\n"
                "DOS version:\n"
                "FreeBASIC version:\n"
                "Storage:\n"
                "Cache/turbo state:\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit) as raised:
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
            self.assertIn("notes_field_empty=Machine key", str(raised.exception))

            with contextlib.redirect_stdout(io.StringIO()):
                written = stage_hardware_capture_evidence.stage_capture(
                    capture,
                    evidence,
                    "486dx2_66_dos622",
                    require_assistant=True,
                    require_notes=True,
                    require_filled_notes=False,
                    force=False,
                )
            self.assertTrue(written)

    def test_stage_capture_rejects_machine_key_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            evidence = root / "evidence"
            capture.mkdir()
            stage_hardware_capture_evidence.write_sample_capture(capture)
            notes = capture / "HWNOTES.TXT"
            notes.write_text(
                notes.read_text(encoding="ascii").replace(
                    "Machine key: 486dx2_66_dos622",
                    "Machine key: 486dx4_100_dos622",
                ),
                encoding="ascii",
            )

            with self.assertRaises(SystemExit) as raised:
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
            self.assertIn("machine_key_mismatch=notes:486dx4_100_dos622", str(raised.exception))

    def test_stage_capture_rejects_unsafe_machine_key(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            stage_hardware_capture_evidence.validate_machine_key("../bad")
        self.assertIn("invalid_machine_key", str(raised.exception))

    def test_stage_capture_self_test(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stage_hardware_capture_evidence.self_test()

        text = output.getvalue()
        self.assertIn("PROBE_OK hardware_capture_staged=486dx2_66_dos622", text)
        self.assertIn("PROBE_OK hardware_stage_self_test=1", text)


if __name__ == "__main__":
    unittest.main()

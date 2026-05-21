from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from scripts import build_dosbox_bundle
from scripts import build_hardware_transfer
from scripts import verify_hardware_capture


ROOT = Path(__file__).resolve().parents[1]


class VerifyHardwareCaptureTests(unittest.TestCase):
    def test_verify_hardware_capture_self_test_artifacts(self) -> None:
        artifact_names = [
            "HWVALID.LOG",
            "QUAL.LOG",
            "PERF.LOG",
            "ASSIST.LOG",
            "ASTRESS.LOG",
            "ASSISTC.LOG",
            "HWNOTES.TXT",
        ]
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

    def test_verify_notes_requires_filled_fields_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            notes = Path(tmp) / "HWNOTES.TXT"
            notes.write_text(
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

            self.assertTrue(
                verify_hardware_capture.verify_notes(
                    notes,
                    required=True,
                    require_values=False,
                )
            )
            with self.assertRaises(SystemExit) as raised:
                verify_hardware_capture.verify_notes(
                    notes,
                    required=True,
                    require_values=True,
                )
            self.assertIn("notes_field_empty=Machine key", str(raised.exception))

    def test_hardware_transfer_self_test(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            build_hardware_transfer.self_test()

        text = output.getvalue()
        self.assertIn("PROBE_OK hardware_transfer_8_3=1", text)
        self.assertIn("PROBE_OK hardware_transfer_readme=README.TXT", text)
        self.assertIn("PROBE_OK hardware_transfer_manifest=1", text)
        self.assertIn("PROBE_OK hardware_transfer_tracked_inputs=1", text)
        self.assertIn("PROBE_OK hardware_transfer_return_checklist=RETURN.TXT", text)
        self.assertIn("PROBE_OK hardware_transfer_zip=1", text)
        self.assertIn("PROBE_OK hardware_transfer_zip_sha256=1", text)

    def test_hardware_transfer_return_instructions_are_dos_friendly(self) -> None:
        text = build_hardware_transfer.return_instructions()

        self.assertIn("HWVALID.LOG", text)
        self.assertIn("QUAL.LOG", text)
        self.assertIn("PERF.LOG", text)
        self.assertIn("ASSIST.LOG", text)
        self.assertIn("ASTRESS.LOG", text)
        self.assertIn("ASSISTC.LOG", text)
        self.assertIn("HWNOTES.TXT", text)
        self.assertIn("--require-filled-notes", text)
        text.encode("ascii")

    def test_hardware_transfer_copied_tree_files_excludes_transient_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "KEEP.TXT").write_text("keep\n", encoding="ascii")
            (root / ".DS_Store").write_text("cache\n", encoding="ascii")
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "MODULE.PYO").write_bytes(b"cache")
            names = {path.relative_to(root).as_posix() for path in build_hardware_transfer.copied_tree_files(root)}

        self.assertEqual(names, {"KEEP.TXT"})

    def test_hardware_transfer_detects_untracked_release_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracked = root / "TRACKED.TXT"
            local = root / "MODEL" / "LOCAL.TXT"
            local.parent.mkdir()
            tracked.write_text("tracked\n", encoding="ascii")
            local.write_text("local\n", encoding="ascii")

            missing = build_hardware_transfer.untracked_release_inputs(
                [tracked, local],
                {"TRACKED.TXT"},
                root=root,
            )

        self.assertEqual(missing, ["MODEL/LOCAL.TXT"])

    def test_dosbox_bundle_self_test(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            build_dosbox_bundle.self_test()

        text = output.getvalue()
        self.assertIn("PROBE_OK dosbox_bundle_profiles=8", text)
        self.assertIn("PROBE_OK dosbox_bundle_relative_mount=1", text)
        self.assertIn("PROBE_OK dosbox_bundle_dpmi_host=1", text)
        self.assertIn("PROBE_OK dosbox_bundle_assistant_shell=1", text)
        self.assertIn("PROBE_OK dosbox_bundle_interactive_profile=1", text)
        self.assertIn("PROBE_OK dosbox_bundle_launchers=1", text)
        self.assertIn("PROBE_OK dosbox_bundle_zip=1", text)
        self.assertIn("PROBE_OK dosbox_bundle_zip_sha256=1", text)

    def test_dosbox_configs_use_relative_mounts(self) -> None:
        for profile in build_dosbox_bundle.DOSBOX_PROFILES:
            text = build_dosbox_bundle.profile_conf(profile)
            self.assertIn("mount c .", text)
            self.assertIn("cd \\GPT2", text)
            self.assertNotIn(str(ROOT), text)
            text.encode("ascii")

    def test_dosbox_bundle_has_interactive_profile(self) -> None:
        profile_names = {profile.name for profile in build_dosbox_bundle.DOSBOX_PROFILES}

        self.assertIn("chat", profile_names)
        self.assertIn("completion", profile_names)

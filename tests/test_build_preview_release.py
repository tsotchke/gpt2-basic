#!/usr/bin/env python3
"""Focused validators for the preview-release package manifest."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_exported_models import AuditRow, ExportedModel  # noqa: E402
from build_preview_release import (  # noqa: E402
    DEFAULT_GENERATED_DATE,
    DEFAULT_ZIP,
    RELEASE_MODELS,
    create_zip,
    file_sha256,
    selected_evidence,
    write_manifest,
)
from build_hardware_transfer import sha256 as hardware_sha256  # noqa: E402
from build_hardware_transfer import write_zip as write_hardware_zip  # noqa: E402
from era_performance_model import Config  # noqa: E402
from verify_preview_artifacts import self_test as preview_artifacts_self_test  # noqa: E402
from verify_preview_artifacts import (  # noqa: E402
    EXPECTED_RELEASE_MODEL_DIRS,
    REQUIRED_DEMO_LOG_MARKERS,
    REQUIRED_MANIFEST_MARKERS,
    REQUIRED_PREVIEW_FILES,
    archive_root_name,
    extract_checked_zip_root,
    read_sidecar,
    require_relative_artifact_path,
    verify_hardware_manifest,
    verify_preview_checksums,
    verify_tree_matches,
    verify_preview_contract,
)


class BuildPreviewReleaseTest(unittest.TestCase):
    def write_minimal_preview_contract(self, root: Path) -> None:
        for model in EXPECTED_RELEASE_MODEL_DIRS:
            model_dir = root / "assets" / "gpt2_basic" / model
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "PROFILE.TXT").write_text("profile\n", encoding="ascii")
        for rel in REQUIRED_PREVIEW_FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if rel == "qemu/evidence/run_main_486.log":
                path.write_text("\n".join(REQUIRED_DEMO_LOG_MARKERS) + "\n", encoding="ascii")
            elif rel == "preview_release_manifest.md":
                path.write_text("\n".join(REQUIRED_MANIFEST_MARKERS) + "\n", encoding="ascii")
            else:
                path.write_text("required\n", encoding="ascii")

    def test_verify_preview_artifacts_self_test(self) -> None:
        preview_artifacts_self_test()

    def test_verify_preview_contract_rejects_deferred_payload_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_minimal_preview_contract(root)
            verify_preview_contract(root)

            deferred = root / "assets" / "media" / "warp4.iso"
            deferred.parent.mkdir(parents=True)
            deferred.write_text("deferred\n", encoding="ascii")

            with self.assertRaises(SystemExit):
                verify_preview_contract(root)

    def test_verify_preview_contract_rejects_transient_payload_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_minimal_preview_contract(root)
            verify_preview_contract(root)

            transient = root / "__pycache__" / "payload.pyc"
            transient.parent.mkdir(parents=True)
            transient.write_bytes(b"cache\n")

            with self.assertRaises(SystemExit):
                verify_preview_contract(root)

    def test_zip_payload_rejects_unsafe_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "unsafe.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("../escape.txt", "bad\n")

            with self.assertRaises(SystemExit):
                extract_checked_zip_root(zip_path, "preview", root / "out")

    def test_zip_payload_rejects_transient_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "transient.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("preview/__pycache__/payload.pyc", "cache\n")

            with self.assertRaises(SystemExit):
                extract_checked_zip_root(zip_path, "preview", root / "out")

    def test_zip_payload_rejects_unsafe_directory_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "unsafe-dir.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("../escape/", "")

            with self.assertRaises(SystemExit):
                extract_checked_zip_root(zip_path, "preview", root / "out")

    def test_zip_payload_rejects_duplicate_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(zip_path, "w") as archive:
                    archive.writestr("preview/payload.txt", "one\n")
                    archive.writestr("preview/payload.txt", "two\n")

            with self.assertRaises(SystemExit):
                extract_checked_zip_root(zip_path, "preview", root / "out")

    def test_zip_payload_rejects_non_deterministic_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "metadata.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("preview/payload.txt", "payload\n")

            with self.assertRaises(SystemExit):
                extract_checked_zip_root(zip_path, "preview", root / "out")

    def test_zip_payload_rejects_non_normalized_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx, member in enumerate(("preview/./payload.txt", "preview//payload.txt", "preview\\payload.txt", "preview/dir/")):
                zip_path = root / f"bad-{idx}.zip"
                with zipfile.ZipFile(zip_path, "w") as archive:
                    archive.writestr(member, "payload\n")

                with self.assertRaises(SystemExit):
                    extract_checked_zip_root(zip_path, "preview", root / f"out-{idx}")

    def test_manifest_paths_require_strict_posix_relative_form(self) -> None:
        for value in ("./payload.txt", "dir//payload.txt", "dir\\payload.txt", "dir/payload.txt\n"):
            with self.assertRaises(SystemExit):
                require_relative_artifact_path(value, "test")

    def test_archive_root_name_handles_dot_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(archive_root_name(Path(".")), root.name)
            finally:
                os.chdir(cwd)

    def test_tree_match_rejects_stale_zip_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live"
            extracted = root / "extracted"
            live.mkdir()
            extracted.mkdir()
            (live / "payload.txt").write_text("new\n", encoding="ascii")
            (extracted / "payload.txt").write_text("old\n", encoding="ascii")

            with self.assertRaises(SystemExit):
                verify_tree_matches(live, extracted, "preview")

    def test_sidecar_rejects_non_sha256_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "artifact.zip.sha256"
            sidecar.write_text("not-a-sha256  artifact.zip\n", encoding="ascii")

            with self.assertRaises(SystemExit):
                read_sidecar(sidecar)

    def test_preview_checksums_reject_escaping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checksum = root / "SHA256SUMS.txt"
            checksum.write_text("0" * 64 + "  ../escape.txt\n", encoding="ascii")

            with self.assertRaises(SystemExit):
                verify_preview_checksums(root)

    def test_preview_checksums_reject_transient_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / ".DS_Store"
            payload.write_text("cache\n", encoding="ascii")
            (root / "SHA256SUMS.txt").write_text(
                f"{file_sha256(payload)}  .DS_Store\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit):
                verify_preview_checksums(root)

    def test_preview_checksums_reject_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "payload.txt"
            payload.write_text("payload\n", encoding="ascii")
            digest = file_sha256(payload)
            (root / "SHA256SUMS.txt").write_text(
                f"{digest}  payload.txt\n{digest}  payload.txt\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit):
                verify_preview_checksums(root)

    def test_preview_checksums_reject_unsorted_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "A.TXT"
            second = root / "B.TXT"
            first.write_text("a\n", encoding="ascii")
            second.write_text("b\n", encoding="ascii")
            (root / "SHA256SUMS.txt").write_text(
                f"{file_sha256(second)}  B.TXT\n{file_sha256(first)}  A.TXT\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit):
                verify_preview_checksums(root)

    def test_hardware_manifest_rejects_non_sha256_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.TXT").write_text("payload\n", encoding="ascii")
            (root / "MANIFEST.TXT").write_text(
                "GPT2-BASIC physical hardware transfer manifest\n"
                "SHA256  BYTES  PATH\n"
                "not-a-sha256  8  README.TXT\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit):
                verify_hardware_manifest(root)

    def test_hardware_manifest_rejects_transient_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "__pycache__" / "BAD.PYO"
            payload.parent.mkdir(parents=True)
            payload.write_text("cache\n", encoding="ascii")
            (root / "MANIFEST.TXT").write_text(
                "GPT2-BASIC physical hardware transfer manifest\n"
                "SHA256  BYTES  PATH\n"
                f"{file_sha256(payload)}  {payload.stat().st_size}  __pycache__/BAD.PYO\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit):
                verify_hardware_manifest(root)

    def test_hardware_manifest_rejects_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "README.TXT"
            payload.write_text("payload\n", encoding="ascii")
            digest = file_sha256(payload)
            (root / "MANIFEST.TXT").write_text(
                "GPT2-BASIC physical hardware transfer manifest\n"
                "SHA256  BYTES  PATH\n"
                f"{digest}  {payload.stat().st_size}  README.TXT\n"
                f"{digest}  {payload.stat().st_size}  README.TXT\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit):
                verify_hardware_manifest(root)

    def test_hardware_manifest_rejects_unsorted_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "A.TXT"
            second = root / "B.TXT"
            first.write_text("a\n", encoding="ascii")
            second.write_text("b\n", encoding="ascii")
            (root / "MANIFEST.TXT").write_text(
                "GPT2-BASIC physical hardware transfer manifest\n"
                "SHA256  BYTES  PATH\n"
                f"{file_sha256(second)}  {second.stat().st_size}  B.TXT\n"
                f"{file_sha256(first)}  {first.stat().st_size}  A.TXT\n",
                encoding="ascii",
            )

            with self.assertRaises(SystemExit):
                verify_hardware_manifest(root)

    def test_preview_zip_is_reproducible_across_mtime_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "preview"
            package.mkdir()
            payload = package / "payload.txt"
            payload.write_text("payload\n", encoding="ascii")

            first_zip = root / "first.zip"
            second_zip = root / "second.zip"
            create_zip(package, first_zip, force=True)
            os.utime(payload, (1, 1))
            create_zip(package, second_zip, force=True)

            self.assertEqual(file_sha256(first_zip), file_sha256(second_zip))

    def test_hardware_zip_is_reproducible_across_mtime_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "hardware"
            package.mkdir()
            payload = package / "PAYLOAD.TXT"
            payload.write_text("payload\n", encoding="ascii")

            first_zip = root / "first.zip"
            second_zip = root / "second.zip"
            write_hardware_zip(package, first_zip, force=True)
            os.utime(payload, (1, 1))
            write_hardware_zip(package, second_zip, force=True)

            self.assertEqual(hardware_sha256(first_zip), hardware_sha256(second_zip))

    def test_manifest_uses_explicit_generated_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "preview"
            package_dir.mkdir()
            cfg = Config(
                profile="test",
                n_layer=2,
                n_embd=48,
                n_head=4,
                n_positions=192,
                hidden_dim=192,
                vocab_size=4096,
            )
            row = AuditRow(
                ExportedModel("MODEL_TEST", "root", ROOT / "assets" / "gpt2_basic" / "MODEL"),
                cfg,
                True,
                ROOT / "qemu" / "evidence" / "model_inventory_model_test.log",
                (),
            )

            manifest = write_manifest(
                rows=[row],
                selected=[(RELEASE_MODELS[0], row)],
                assistants=[],
                evidence_files=[],
                output_dir=package_dir,
                zip_path=DEFAULT_ZIP,
                package_built=False,
                generated_date="1999-12-31",
            )

        self.assertIn("Generated: `1999-12-31`", manifest)

    def test_selected_evidence_includes_dos_demo_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp)
            (evidence_dir / "run_main_486.log").write_text("demo\n", encoding="ascii")
            (evidence_dir / "scratch.log").write_text("ignore\n", encoding="ascii")

            names = {path.name for path in selected_evidence(evidence_dir)}

        self.assertIn("run_main_486.log", names)
        self.assertNotIn("scratch.log", names)

    def test_manifest_reports_existing_package_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "preview"
            package_dir.mkdir()
            (package_dir / "A.TXT").write_text("abc\n", encoding="ascii")
            (package_dir / "B.TXT").write_text("de\n", encoding="ascii")

            cfg = Config(
                profile="test",
                n_layer=2,
                n_embd=48,
                n_head=4,
                n_positions=192,
                hidden_dim=192,
                vocab_size=4096,
            )
            row = AuditRow(
                ExportedModel("MODEL_TEST", "root", ROOT / "assets" / "gpt2_basic" / "MODEL"),
                cfg,
                True,
                ROOT / "qemu" / "evidence" / "model_inventory_model_test.log",
                (),
            )

            manifest = write_manifest(
                rows=[row],
                selected=[(RELEASE_MODELS[0], row)],
                assistants=[],
                evidence_files=[ROOT / "qemu" / "evidence" / "run_main_486.log"],
                output_dir=package_dir,
                zip_path=DEFAULT_ZIP,
                package_built=True,
                generated_date=DEFAULT_GENERATED_DATE,
            )

        self.assertIn("Package status: `2 files, 7 bytes`", manifest)
        self.assertIn("qemu/evidence/run_main_486.log", manifest)


if __name__ == "__main__":
    unittest.main()

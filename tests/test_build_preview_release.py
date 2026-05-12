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
    copied_tree_files,
    create_zip,
    file_sha256,
    selected_evidence,
    untracked_release_inputs,
    write_converged_package_manifest,
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
    verify_no_host_absolute_paths,
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

    def test_verify_preview_contract_requires_workspace_tracking_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_minimal_preview_contract(root)
            probe = root / "qemu" / "evidence" / "workspace_tracking_probe.log"
            probe.unlink()

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

    def test_preview_tree_rejects_host_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "qemu" / "evidence" / "report.md"
            payload.parent.mkdir(parents=True)
            forbidden = "/" + "Users/example/project/report.log"
            payload.write_text(f"Source: {forbidden}\n", encoding="ascii")

            with self.assertRaises(SystemExit):
                verify_no_host_absolute_paths(root, "preview")

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

    def test_manifest_uses_portable_artifact_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "preview"
            package_dir.mkdir()
            zip_path = Path(tmp) / "preview.zip"
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
                zip_path=zip_path,
                package_built=False,
                generated_date=DEFAULT_GENERATED_DATE,
            )

        self.assertIn("Package tree: `preview`", manifest)
        self.assertIn("Package zip: `preview.zip`", manifest)
        self.assertNotIn(str(package_dir.parent), manifest)

    def test_converged_manifest_ignores_stale_output_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            manifests: list[str] = []
            for parent_name, stale_text in (("a", "old\n"), ("b", "old output path /tmp/build/with/a/long/name\n")):
                package_dir = root / parent_name / "preview"
                package_dir.mkdir(parents=True)
                (package_dir / "payload.txt").write_text("payload\n", encoding="ascii")
                (package_dir / "preview_release_manifest.md").write_text(stale_text, encoding="ascii")
                manifests.append(
                    write_converged_package_manifest(
                        rows=[row],
                        selected=[(RELEASE_MODELS[0], row)],
                        assistants=[],
                        evidence=[],
                        output_dir=package_dir,
                        zip_path=root / parent_name / "preview.zip",
                        manifest_path=root / parent_name / "manifest.md",
                        generated_date=DEFAULT_GENERATED_DATE,
                    )
                )

            self.assertEqual(manifests[0], manifests[1])
            self.assertIn("Package status: `3 files", manifests[0])

    def test_selected_evidence_includes_dos_demo_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp)
            (evidence_dir / "run_main_486.log").write_text("demo\n", encoding="ascii")
            (evidence_dir / "scratch.log").write_text("ignore\n", encoding="ascii")

            names = {path.name for path in selected_evidence(evidence_dir)}

        self.assertIn("run_main_486.log", names)
        self.assertNotIn("scratch.log", names)

    def test_copied_tree_files_excludes_transient_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.txt").write_text("keep\n", encoding="ascii")
            (root / ".DS_Store").write_text("cache\n", encoding="ascii")
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "module.pyc").write_bytes(b"cache")
            names = {path.relative_to(root).as_posix() for path in copied_tree_files(root)}

        self.assertEqual(names, {"keep.txt"})

    def test_untracked_release_inputs_rejects_local_package_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracked = root / "tracked.txt"
            untracked = root / "data" / "domain_curriculum" / "local.txt"
            outside = Path(tempfile.gettempdir()) / "outside-preview-input.txt"
            untracked.parent.mkdir(parents=True)
            tracked.write_text("tracked\n", encoding="ascii")
            untracked.write_text("local\n", encoding="ascii")
            outside.write_text("outside\n", encoding="ascii")
            try:
                missing = untracked_release_inputs(
                    [tracked, untracked, outside],
                    {"tracked.txt"},
                    root=root,
                )
            finally:
                outside.unlink(missing_ok=True)

        self.assertEqual(missing, ["data/domain_curriculum/local.txt"])

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
        self.assertIn("## Verification Commands", manifest)
        self.assertIn("python3 scripts/verify_workspace_tracking.py", manifest)
        self.assertIn("python3 scripts/verify_preview_artifacts.py", manifest)


if __name__ == "__main__":
    unittest.main()

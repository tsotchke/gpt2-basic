#!/usr/bin/env python3
"""Verify GPT2-BASIC preview-release package artifacts."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path


DEFAULT_PREVIEW_DIR = Path("/private/tmp/gpt2-basic-preview")
DEFAULT_PREVIEW_ZIP = Path("/private/tmp/gpt2-basic-preview.zip")
DEFAULT_PREVIEW_ZIP_SHA256 = Path("/private/tmp/gpt2-basic-preview.zip.sha256")
DEFAULT_HARDWARE_DIR = Path("/private/tmp/gpt2-basic-hardware-transfer")
DEFAULT_HARDWARE_ZIP = Path("/private/tmp/gpt2-basic-hardware-transfer.zip")
DEFAULT_HARDWARE_ZIP_SHA256 = Path("/private/tmp/gpt2-basic-hardware-transfer.zip.sha256")
DEFAULT_REPO_MANIFEST = Path(__file__).resolve().parents[1] / "qemu" / "evidence" / "preview_release_manifest.md"
DETERMINISTIC_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
DETERMINISTIC_ZIP_CREATE_SYSTEM = 3
DETERMINISTIC_ZIP_EXTERNAL_ATTR = 0o100644 << 16
DETERMINISTIC_ZIP_COMPRESS_TYPE = zipfile.ZIP_DEFLATED
SHA256_HEX = set("0123456789abcdef")
FORBIDDEN_TRANSIENT_NAMES = {"__pycache__", ".ds_store"}
FORBIDDEN_TRANSIENT_SUFFIXES = (".pyc", ".pyo")
FORBIDDEN_HOST_PATH_FRAGMENTS = (
    bytes((47,)) + b"Users" + bytes((47,)),
    bytes((92,)) + b"Users" + bytes((92,)),
)

EXPECTED_RELEASE_MODEL_DIRS = {
    "MODEL",
    "MODEL_LEXICON_GOLD_V4_S3000",
    "MODEL_HEADSHORTLIST2048_PROD_PROBE",
    "MODEL_HEADQ4_PROD_PROBE",
    "MODEL_TOKHEADQ4_PROD_PROBE",
    "MODEL_TOKHEADQ4_STREAM_PROD_PROBE",
}
REQUIRED_PREVIEW_FILES = (
    "bin/GPT2.EXE",
    "docs/dosbox.md",
    "docs/releases/v0.1.0-preview.md",
    "assets/gpt2_basic/MODEL/VOCAB.BIN",
    "assets/gpt2_basic/PACKS/CHAT/PACK.INI",
    "assets/gpt2_basic/PACKS/CHAT/USAGE.TXT",
    "assets/gpt2_basic/PACKS/CHAT/KDBIDX.TXT",
    "assets/gpt2_basic/PACKS/DOSHELP/PACK.INI",
    "assets/gpt2_basic/PACKS/DOSHELP/USAGE.TXT",
    "assets/gpt2_basic/PACKS/DOSHELP/KDBIDX.TXT",
    "assets/gpt2_basic/PACKS/OFFICE/PACK.INI",
    "assets/gpt2_basic/PACKS/OFFICE/USAGE.TXT",
    "assets/gpt2_basic/PACKS/OFFICE/KDBIDX.TXT",
    "assets/gpt2_basic/PACKS/DEV/PACK.INI",
    "assets/gpt2_basic/PACKS/DEV/USAGE.TXT",
    "assets/gpt2_basic/PACKS/DEV/KDBIDX.TXT",
    "qemu/evidence/run_main_486.log",
    "qemu/evidence/workspace_tracking_probe.log",
    "scripts/build_dosbox_bundle.py",
    "tests/test_build_preview_release.py",
    "tests/test_workspace_tracking.py",
    "preview_release_manifest.md",
    "SHA256SUMS.txt",
)
FORBIDDEN_PREVIEW_PATH_FRAGMENTS = (
    "assets/media/",
    "vm/",
    "86box",
    "os2warp",
    "warp4",
)
REQUIRED_DEMO_LOG_MARKERS = (
    "GPT2-BASIC production trained-model demo",
    "Loaded vocabulary with  4096 tokens",
    "Generated  35 tokens",
    "Runtime memory:  2055940 bytes",
)
REQUIRED_MANIFEST_MARKERS = (
    "This preview release is the DOS demo, DOSBox convenience package, and DOS transfer package.",
    "OS/2/Warp package, stay on the later-release track.",
    "scripts/build_dosbox_bundle.py",
    "CHAT, DOSHELP, OFFICE, and DEV packs",
    "generated `KDB.TXT`",
    "`KDBIDX.TXT` recall files",
    "per-pack `USAGE.TXT`",
    "qemu/evidence/run_main_486.log",
    "qemu/evidence/workspace_tracking_probe.log",
)
REQUIRED_HARDWARE_FILES = (
    "GPT2/GPT2.EXE",
    "GPT2/HWVALID.BAT",
    "GPT2/HWNOTES.TXT",
    "GPT2/RETURN.TXT",
    "GPT2/MODEL/VOCAB.BIN",
    "GPT2/PACKS/CHAT/PACK.INI",
    "GPT2/PACKS/CHAT/USAGE.TXT",
    "GPT2/PACKS/CHAT/KDBIDX.TXT",
    "GPT2/PACKS/DOSHELP/PACK.INI",
    "GPT2/PACKS/DOSHELP/USAGE.TXT",
    "GPT2/PACKS/DOSHELP/KDBIDX.TXT",
    "GPT2/PACKS/OFFICE/PACK.INI",
    "GPT2/PACKS/OFFICE/USAGE.TXT",
    "GPT2/PACKS/OFFICE/KDBIDX.TXT",
    "GPT2/PACKS/DEV/PACK.INI",
    "GPT2/PACKS/DEV/USAGE.TXT",
    "GPT2/PACKS/DEV/KDBIDX.TXT",
    "GPT2/GPT2SRC/MAIN.BAS",
    "README.TXT",
    "MANIFEST.TXT",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PREVIEW_ARTIFACTS_FAILED {message}")


def require_sha256_hex(value: str, label: str) -> None:
    require(len(value) == 64 and all(ch in SHA256_HEX for ch in value), f"bad_sha256_{label}={value}")


def require_relative_artifact_path(value: str, label: str) -> None:
    parts = value.split("/")
    require(value and "\\" not in value, f"bad_artifact_path_{label}={value}")
    require(all(ord(ch) >= 32 for ch in value), f"bad_artifact_path_{label}={value}")
    require(
        not value.startswith("/") and all(part and part not in {".", ".."} for part in parts),
        f"bad_artifact_path_{label}={value}",
    )


def require_no_transient_artifact_path(value: str, label: str) -> None:
    parts = [part.lower() for part in value.split("/")]
    require(not any(part in FORBIDDEN_TRANSIENT_NAMES for part in parts), f"{label}_transient_artifact={value}")
    require(
        not any(parts[-1].endswith(suffix) for suffix in FORBIDDEN_TRANSIENT_SUFFIXES),
        f"{label}_transient_artifact={value}",
    )


def verify_no_transient_artifacts(root: Path, label: str) -> None:
    for path in root.rglob("*"):
        require_no_transient_artifact_path(path.relative_to(root).as_posix(), label)


def verify_no_host_absolute_paths(root: Path, label: str) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        for fragment in FORBIDDEN_HOST_PATH_FRAGMENTS:
            require(fragment not in data, f"{label}_host_path_leak={rel}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def verify_tree_matches(left: Path, right: Path, label: str) -> None:
    left_hashes = file_hashes(left)
    right_hashes = file_hashes(right)
    left_names = set(left_hashes)
    right_names = set(right_hashes)
    missing = sorted(left_names - right_names)
    extra = sorted(right_names - left_names)
    changed = sorted(name for name in left_names & right_names if left_hashes[name] != right_hashes[name])
    require(not missing, f"{label}_zip_missing=" + ",".join(missing[:5]))
    require(not extra, f"{label}_zip_extra=" + ",".join(extra[:5]))
    require(not changed, f"{label}_zip_mismatch=" + ",".join(changed[:5]))


def require_unique(value: str, seen: set[str], label: str) -> None:
    require(value not in seen, f"duplicate_{label}={value}")
    seen.add(value)


def require_sorted(value: str, previous: str | None, label: str) -> str:
    require(previous is None or previous < value, f"unsorted_{label}={value}")
    return value


def read_sidecar(path: Path) -> tuple[str, str]:
    require(path.is_file(), f"missing_sidecar={path}")
    parts = path.read_text(encoding="ascii").strip().split()
    require(len(parts) == 2, f"bad_sidecar={path}")
    require_sha256_hex(parts[0], f"sidecar:{path}")
    return parts[0], parts[1]


def verify_zip(zip_path: Path, sidecar_path: Path) -> None:
    require(zip_path.is_file(), f"missing_zip={zip_path}")
    expected, name = read_sidecar(sidecar_path)
    require(name == zip_path.name, f"sidecar_name={name} expected={zip_path.name}")
    require(sha256(zip_path) == expected, f"zip_sha256={zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
    require(bad is None, f"zip_integrity={zip_path}:{bad}")


def extract_checked_zip_root(zip_path: Path, expected_root: str, dest_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        infos = [info for info in archive.infolist() if info.filename]
        all_members = [info.filename for info in infos]
        file_members = [info.filename for info in infos if not info.is_dir()]
        require(file_members, f"zip_empty={zip_path}")
        seen_members: set[str] = set()
        for info in infos:
            member = info.filename
            require_unique(member, seen_members, "zip_member")
            require(not info.is_dir(), f"zip_directory_member={member}")
            require_relative_artifact_path(member, "zip_member")
            require_no_transient_artifact_path(member, "zip_member")
            require(info.date_time == DETERMINISTIC_ZIP_TIMESTAMP, f"zip_timestamp={member}:{info.date_time}")
            require(info.compress_type == DETERMINISTIC_ZIP_COMPRESS_TYPE, f"zip_compression={member}:{info.compress_type}")
            require(info.create_system == DETERMINISTIC_ZIP_CREATE_SYSTEM, f"zip_create_system={member}:{info.create_system}")
            require(info.external_attr == DETERMINISTIC_ZIP_EXTERNAL_ATTR, f"zip_external_attr={member}:{oct(info.external_attr >> 16)}")
            parts = member.split("/")
            require(parts[0] == expected_root, f"zip_root={member} expected={expected_root}")
        archive.extractall(dest_dir)
    root = dest_dir / expected_root
    require(root.is_dir(), f"zip_root_missing={root}")
    return root


def verify_preview_zip_payload(zip_path: Path, expected_root: str, repo_manifest: Path | None, source_dir: Path | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        extracted = extract_checked_zip_root(zip_path, expected_root, Path(tmp))
        verify_preview_tree(extracted, repo_manifest)
        if source_dir is not None:
            verify_tree_matches(source_dir, extracted, "preview")


def verify_hardware_zip_payload(zip_path: Path, expected_root: str, source_dir: Path | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        extracted = extract_checked_zip_root(zip_path, expected_root, Path(tmp))
        verify_hardware_manifest(extracted)
        if source_dir is not None:
            verify_tree_matches(source_dir, extracted, "hardware")


def verify_preview_checksums(preview_dir: Path) -> None:
    checksum_path = preview_dir / "SHA256SUMS.txt"
    require(checksum_path.is_file(), f"missing_preview_checksums={checksum_path}")
    seen: set[str] = set()
    previous_rel: str | None = None
    for lineno, line in enumerate(checksum_path.read_text(encoding="ascii").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        require(len(parts) == 2, f"bad_preview_checksum_line={lineno}")
        expected, rel = parts
        require_sha256_hex(expected, f"preview:{lineno}")
        rel = rel.strip()
        require_relative_artifact_path(rel, f"preview:{lineno}")
        require_no_transient_artifact_path(rel, f"preview:{lineno}")
        require_unique(rel, seen, "preview_checksum")
        previous_rel = require_sorted(rel, previous_rel, "preview_checksum")
        path = preview_dir / rel
        require(path.is_file(), f"preview_checksum_missing={lineno}:{rel}")
        require(sha256(path) == expected, f"preview_checksum_mismatch={rel}")
    expected_paths = {
        path.relative_to(preview_dir).as_posix()
        for path in preview_dir.rglob("*")
        if path.is_file() and path != checksum_path
    }
    missing = sorted(expected_paths - seen)
    extra = sorted(seen - expected_paths)
    require(not missing, "preview_checksum_unlisted=" + ",".join(missing[:5]))
    require(not extra, "preview_checksum_extra=" + ",".join(extra[:5]))


def manifest_status(manifest: Path) -> str:
    for line in manifest.read_text(encoding="ascii").splitlines():
        if line.startswith("Package status:"):
            parts = line.split("`")
            require(len(parts) >= 3, f"bad_package_status={manifest}")
            return parts[1]
    raise SystemExit(f"PREVIEW_ARTIFACTS_FAILED missing_package_status={manifest}")


def dir_status(path: Path) -> str:
    files = [item for item in path.rglob("*") if item.is_file()]
    return f"{len(files)} files, {sum(item.stat().st_size for item in files):,} bytes"


def archive_root_name(path: Path) -> str:
    return path.resolve().name


def verify_preview_contract(preview_dir: Path) -> None:
    for rel in REQUIRED_PREVIEW_FILES:
        require((preview_dir / rel).is_file(), f"preview_required_missing={rel}")

    paths = [path.relative_to(preview_dir).as_posix() for path in preview_dir.rglob("*") if path.is_file()]
    for rel in paths:
        require_no_transient_artifact_path(rel, "preview")
        lower = rel.lower()
        for fragment in FORBIDDEN_PREVIEW_PATH_FRAGMENTS:
            require(fragment not in lower, f"preview_deferred_payload={rel}")

    model_root = preview_dir / "assets" / "gpt2_basic"
    require(model_root.is_dir(), f"missing_model_root={model_root}")
    actual_models = {path.name for path in model_root.iterdir() if path.is_dir() and path.name.startswith("MODEL")}
    require(actual_models == EXPECTED_RELEASE_MODEL_DIRS, "preview_release_model_set=" + ",".join(sorted(actual_models)))

    demo_log = (preview_dir / "qemu" / "evidence" / "run_main_486.log").read_text(encoding="ascii")
    for marker in REQUIRED_DEMO_LOG_MARKERS:
        require(marker in demo_log, f"preview_demo_log_marker={marker}")

    manifest_text = (preview_dir / "preview_release_manifest.md").read_text(encoding="ascii")
    for marker in REQUIRED_MANIFEST_MARKERS:
        require(marker in manifest_text, f"preview_manifest_marker={marker}")


def verify_preview_tree(preview_dir: Path, repo_manifest: Path | None) -> None:
    require(preview_dir.is_dir(), f"missing_preview_dir={preview_dir}")
    verify_no_transient_artifacts(preview_dir, "preview")
    verify_no_host_absolute_paths(preview_dir, "preview")
    manifest = preview_dir / "preview_release_manifest.md"
    require(manifest.is_file(), f"missing_preview_manifest={manifest}")
    require(manifest_status(manifest) == dir_status(preview_dir), f"preview_manifest_status={manifest}")
    verify_preview_checksums(preview_dir)
    verify_preview_contract(preview_dir)
    if repo_manifest is not None and repo_manifest.exists():
        require(repo_manifest.read_bytes() == manifest.read_bytes(), "repo_manifest_differs_from_package")


def short_name_ok(path: Path) -> bool:
    name = path.name
    base, _dot, ext = name.partition(".")
    if not base or len(base) > 8 or len(ext) > 3 or "." in ext:
        return False
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$~!#%&-{}()@'`"
    return all(ch in allowed for ch in (base + ext).upper())


def verify_hardware_contract(hardware_dir: Path) -> None:
    for rel in REQUIRED_HARDWARE_FILES:
        require((hardware_dir / rel).is_file(), f"hardware_required_missing={rel}")
    dos_root = hardware_dir / "GPT2"
    require(dos_root.is_dir(), f"hardware_missing_dos_root={dos_root}")
    for path in dos_root.rglob("*"):
        require(short_name_ok(path), f"hardware_non_8_3={path.relative_to(dos_root).as_posix()}")


def verify_hardware_manifest(hardware_dir: Path) -> None:
    manifest = hardware_dir / "MANIFEST.TXT"
    require(manifest.is_file(), f"missing_hardware_manifest={manifest}")
    verify_no_transient_artifacts(hardware_dir, "hardware")
    seen: set[str] = set()
    previous_rel: str | None = None
    for lineno, line in enumerate(manifest.read_text(encoding="ascii").splitlines(), start=1):
        if not line or line.startswith(("GPT2-BASIC", "Copy ", "Run ", "SHA256")):
            continue
        parts = line.split("  ", 2)
        require(len(parts) == 3, f"bad_hardware_manifest_line={lineno}")
        expected, size_text, rel = parts
        require_sha256_hex(expected, f"hardware:{lineno}")
        require(size_text.isdecimal(), f"bad_hardware_manifest_size={lineno}:{size_text}")
        require_relative_artifact_path(rel, f"hardware:{lineno}")
        require_no_transient_artifact_path(rel, f"hardware:{lineno}")
        require_unique(rel, seen, "hardware_manifest")
        previous_rel = require_sorted(rel, previous_rel, "hardware_manifest")
        path = hardware_dir / rel
        require(path.is_file(), f"hardware_manifest_missing={rel}")
        require(sha256(path) == expected, f"hardware_manifest_sha256={rel}")
        require(path.stat().st_size == int(size_text), f"hardware_manifest_size={rel}")
    verify_hardware_contract(hardware_dir)


def verify_all(args: argparse.Namespace) -> None:
    if not args.skip_preview:
        verify_preview_tree(args.preview_dir, args.repo_manifest)
        verify_zip(args.preview_zip, args.preview_zip_sha256)
        verify_preview_zip_payload(args.preview_zip, archive_root_name(args.preview_dir), args.repo_manifest, args.preview_dir)
        print("PROBE_OK preview_artifacts_preview_tree=1")
        print("PROBE_OK preview_artifacts_preview_zip=1")
        print("PROBE_OK preview_artifacts_preview_zip_payload=1")
        print("PROBE_OK preview_artifacts_preview_zip_matches_tree=1")
    if not args.skip_hardware:
        verify_hardware_manifest(args.hardware_dir)
        verify_zip(args.hardware_zip, args.hardware_zip_sha256)
        verify_hardware_zip_payload(args.hardware_zip, archive_root_name(args.hardware_dir), args.hardware_dir)
        print("PROBE_OK preview_artifacts_hardware_tree=1")
        print("PROBE_OK preview_artifacts_hardware_zip=1")
        print("PROBE_OK preview_artifacts_hardware_zip_payload=1")
        print("PROBE_OK preview_artifacts_hardware_zip_matches_tree=1")
    require(not (args.skip_preview and args.skip_hardware), "nothing_to_verify")


def write_zip(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(root.parent).as_posix(), date_time=DETERMINISTIC_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        preview = root / "preview"
        preview.mkdir()
        (preview / "payload.txt").write_text("payload\n", encoding="ascii")
        for model in EXPECTED_RELEASE_MODEL_DIRS:
            model_dir = preview / "assets" / "gpt2_basic" / model
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "PROFILE.TXT").write_text("profile\n", encoding="ascii")
        for rel in (
            "bin/GPT2.EXE",
            "docs/dosbox.md",
            "docs/releases/v0.1.0-preview.md",
            "assets/gpt2_basic/MODEL/VOCAB.BIN",
            "assets/gpt2_basic/PACKS/CHAT/PACK.INI",
            "assets/gpt2_basic/PACKS/CHAT/USAGE.TXT",
            "assets/gpt2_basic/PACKS/CHAT/KDBIDX.TXT",
            "assets/gpt2_basic/PACKS/DOSHELP/PACK.INI",
            "assets/gpt2_basic/PACKS/DOSHELP/USAGE.TXT",
            "assets/gpt2_basic/PACKS/DOSHELP/KDBIDX.TXT",
            "assets/gpt2_basic/PACKS/OFFICE/PACK.INI",
            "assets/gpt2_basic/PACKS/OFFICE/USAGE.TXT",
            "assets/gpt2_basic/PACKS/OFFICE/KDBIDX.TXT",
            "assets/gpt2_basic/PACKS/DEV/PACK.INI",
            "assets/gpt2_basic/PACKS/DEV/USAGE.TXT",
            "assets/gpt2_basic/PACKS/DEV/KDBIDX.TXT",
            "scripts/build_dosbox_bundle.py",
            "tests/test_build_preview_release.py",
            "tests/test_workspace_tracking.py",
        ):
            path = preview / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("required\n", encoding="ascii")
        demo_log = preview / "qemu" / "evidence" / "run_main_486.log"
        demo_log.parent.mkdir(parents=True, exist_ok=True)
        demo_log.write_text("\n".join(REQUIRED_DEMO_LOG_MARKERS) + "\n", encoding="ascii")
        workspace_probe = preview / "qemu" / "evidence" / "workspace_tracking_probe.log"
        workspace_probe.write_text("PROBE_OK workspace_tracking_no_untracked=1\n", encoding="ascii")
        preview_manifest = preview / "preview_release_manifest.md"
        status = "pending"
        for _ in range(10):
            preview_manifest.write_text(
                f"Package status: `{status}`\n" + "\n".join(REQUIRED_MANIFEST_MARKERS) + "\n",
                encoding="ascii",
            )
            checksums = [
                f"{sha256(path)}  {path.relative_to(preview).as_posix()}"
                for path in sorted(preview.rglob("*"))
                if path.is_file() and path.name != "SHA256SUMS.txt"
            ]
            (preview / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")
            next_status = dir_status(preview)
            if next_status == status:
                break
            status = next_status
        else:
            raise RuntimeError("self-test preview manifest status did not converge")
        repo_manifest = root / "repo_manifest.md"
        repo_manifest.write_bytes(preview_manifest.read_bytes())
        preview_zip = root / "preview.zip"
        write_zip(preview, preview_zip)
        preview_zip_sha = root / "preview.zip.sha256"
        preview_zip_sha.write_text(f"{sha256(preview_zip)}  {preview_zip.name}\n", encoding="ascii")

        hardware = root / "hardware"
        hardware.mkdir()
        for rel in REQUIRED_HARDWARE_FILES:
            if rel == "MANIFEST.TXT":
                continue
            path = hardware / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("copy\n", encoding="ascii")
        hardware_manifest = hardware / "MANIFEST.TXT"
        hardware_lines = [
            "GPT2-BASIC physical hardware transfer manifest",
            "SHA256  BYTES  PATH",
        ]
        for path in sorted(hardware.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.TXT":
                hardware_lines.append(f"{sha256(path)}  {path.stat().st_size}  {path.relative_to(hardware).as_posix()}")
        hardware_manifest.write_text(
            "\n".join(hardware_lines) + "\n",
            encoding="ascii",
        )
        hardware_zip = root / "hardware.zip"
        write_zip(hardware, hardware_zip)
        hardware_zip_sha = root / "hardware.zip.sha256"
        hardware_zip_sha.write_text(f"{sha256(hardware_zip)}  {hardware_zip.name}\n", encoding="ascii")

        args = argparse.Namespace(
            preview_dir=preview,
            preview_zip=preview_zip,
            preview_zip_sha256=preview_zip_sha,
            hardware_dir=hardware,
            hardware_zip=hardware_zip,
            hardware_zip_sha256=hardware_zip_sha,
            repo_manifest=repo_manifest,
            skip_preview=False,
            skip_hardware=False,
        )
        verify_all(args)
        leaked = root / "leaked"
        leaked.mkdir()
        forbidden = "/" + "Users/example/project/report.log"
        (leaked / "report.md").write_text(f"Source: {forbidden}\n", encoding="ascii")
        try:
            verify_no_host_absolute_paths(leaked, "preview")
        except SystemExit:
            pass
        else:
            raise RuntimeError("self-test did not reject host absolute path leak")
    print("PROBE_OK preview_artifacts_workspace_tracking_probe_required=1")
    print("PROBE_OK preview_artifacts_host_path_rejection=1")
    print("PROBE_OK preview_artifacts_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--preview-zip", type=Path, default=DEFAULT_PREVIEW_ZIP)
    parser.add_argument("--preview-zip-sha256", type=Path, default=DEFAULT_PREVIEW_ZIP_SHA256)
    parser.add_argument("--hardware-dir", type=Path, default=DEFAULT_HARDWARE_DIR)
    parser.add_argument("--hardware-zip", type=Path, default=DEFAULT_HARDWARE_ZIP)
    parser.add_argument("--hardware-zip-sha256", type=Path, default=DEFAULT_HARDWARE_ZIP_SHA256)
    parser.add_argument("--repo-manifest", type=Path, default=DEFAULT_REPO_MANIFEST)
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        verify_all(args)


if __name__ == "__main__":
    main()

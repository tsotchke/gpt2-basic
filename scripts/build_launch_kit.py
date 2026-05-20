#!/usr/bin/env python3
"""Build a launch handoff kit from verified release and promo assets."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_VERSION = "v0.1.0-preview"
DEFAULT_RELEASE_URL = "https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview"
DEFAULT_GENERATED_DATE = "2026-05-19"
DEFAULT_OUTPUT_DIR = Path("/private/tmp/gpt2-basic-launch-kit")
DEFAULT_ZIP = Path("/private/tmp/gpt2-basic-launch-kit.zip")
DEFAULT_ZIP_SHA256 = Path("/private/tmp/gpt2-basic-launch-kit.zip.sha256")
DETERMINISTIC_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
SHA256_HEX = set("0123456789abcdef")


@dataclass(frozen=True)
class KitAsset:
    source: Path
    target: str
    required: bool = True


RELEASE_ASSETS = (
    KitAsset(Path("/private/tmp/gpt2-basic-preview.zip"), "release/gpt2-basic-preview.zip"),
    KitAsset(Path("/private/tmp/gpt2-basic-preview.zip.sha256"), "release/gpt2-basic-preview.zip.sha256"),
    KitAsset(Path("/private/tmp/gpt2-basic-dosbox.zip"), "release/gpt2-basic-dosbox.zip"),
    KitAsset(Path("/private/tmp/gpt2-basic-dosbox.zip.sha256"), "release/gpt2-basic-dosbox.zip.sha256"),
    KitAsset(Path("/private/tmp/gpt2-basic-hardware-transfer.zip"), "release/gpt2-basic-hardware-transfer.zip"),
    KitAsset(
        Path("/private/tmp/gpt2-basic-hardware-transfer.zip.sha256"),
        "release/gpt2-basic-hardware-transfer.zip.sha256",
    ),
    KitAsset(ROOT / "qemu/evidence/preview_release_manifest.md", "release/preview_release_manifest.md"),
)

MEDIA_ASSETS = (
    KitAsset(
        ROOT / "promo/renders/gpt2_basic_assistant_showcase_1080p.mp4",
        "media/gpt2_basic_assistant_showcase_1080p.mp4",
    ),
    KitAsset(ROOT / "promo/renders/gpt2_basic_real_dos_session_1080p.mp4", "media/gpt2_basic_real_dos_session_1080p.mp4"),
    KitAsset(
        ROOT / "promo/renders/gpt2_basic_real_dos_session_vertical.mp4",
        "media/gpt2_basic_real_dos_session_vertical.mp4",
    ),
    KitAsset(ROOT / "promo/renders/gpt2_basic_terminal_demo_1080p.mp4", "media/gpt2_basic_terminal_demo_1080p.mp4"),
    KitAsset(
        ROOT / "promo/renders/gpt2_basic_terminal_demo_vertical.mp4",
        "media/gpt2_basic_terminal_demo_vertical.mp4",
    ),
    KitAsset(ROOT / "promo/renders/gpt2_basic_launch_teaser_1080p.mp4", "media/gpt2_basic_launch_teaser_1080p.mp4"),
    KitAsset(ROOT / "promo/renders/gpt2_basic_launch_short_vertical.mp4", "media/gpt2_basic_launch_short_vertical.mp4"),
    KitAsset(ROOT / "promo/renders/thumbnail_gpt_in_dos.png", "media/thumbnail_gpt_in_dos.png"),
)

COPY_ASSETS = (
    KitAsset(ROOT / "README.md", "copy/README.md"),
    KitAsset(ROOT / "CONTRIBUTING.md", "copy/CONTRIBUTING.md"),
    KitAsset(ROOT / "SECURITY.md", "copy/SECURITY.md"),
    KitAsset(ROOT / "docs/dosbox.md", "copy/dosbox.md"),
    KitAsset(ROOT / "docs/public-launch.md", "copy/public-launch.md"),
    KitAsset(ROOT / "docs/marketing/promo-kit.md", "copy/promo-kit.md"),
    KitAsset(ROOT / "docs/marketing/video-plan.md", "copy/video-plan.md"),
    KitAsset(ROOT / "docs/marketing/public-demo-script.md", "copy/public-demo-script.md"),
    KitAsset(ROOT / "docs/releases/v0.1.0-preview.md", "copy/release-notes.md"),
)

RELEASE_ZIP_SIDECARS = (
    (Path("/private/tmp/gpt2-basic-preview.zip"), Path("/private/tmp/gpt2-basic-preview.zip.sha256")),
    (Path("/private/tmp/gpt2-basic-dosbox.zip"), Path("/private/tmp/gpt2-basic-dosbox.zip.sha256")),
    (
        Path("/private/tmp/gpt2-basic-hardware-transfer.zip"),
        Path("/private/tmp/gpt2-basic-hardware-transfer.zip.sha256"),
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"LAUNCH_KIT_FAILED {message}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip_checksum(zip_path: Path, zip_sha256: Path) -> None:
    zip_sha256.parent.mkdir(parents=True, exist_ok=True)
    zip_sha256.write_text(f"{file_sha256(zip_path)}  {zip_path.name}\n", encoding="ascii")


def read_sidecar(path: Path) -> tuple[str, str]:
    require(path.is_file(), f"missing_sidecar={path}")
    parts = path.read_text(encoding="ascii").strip().split()
    require(len(parts) == 2, f"bad_sidecar={path}")
    require(len(parts[0]) == 64 and all(ch in SHA256_HEX for ch in parts[0]), f"bad_sidecar_sha256={path}")
    return parts[0], parts[1]


def verify_zip_sidecar(zip_path: Path, sidecar_path: Path) -> None:
    require(zip_path.is_file(), f"missing_zip={zip_path}")
    expected, name = read_sidecar(sidecar_path)
    require(name == zip_path.name, f"sidecar_name={name} expected={zip_path.name}")
    require(file_sha256(zip_path) == expected, f"sidecar_sha256_mismatch={zip_path}")


def write_deterministic_zip_entry(archive: zipfile.ZipFile, root_parent: Path, path: Path) -> None:
    info = zipfile.ZipInfo(path.relative_to(root_parent).as_posix(), date_time=DETERMINISTIC_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, path.read_bytes())


def all_assets() -> tuple[KitAsset, ...]:
    return RELEASE_ASSETS + MEDIA_ASSETS + COPY_ASSETS


def verify_release_sidecars() -> None:
    for zip_path, sidecar_path in RELEASE_ZIP_SIDECARS:
        verify_zip_sidecar(zip_path, sidecar_path)


def copy_asset(asset: KitAsset, output_dir: Path) -> Path | None:
    if not asset.source.exists():
        require(not asset.required, f"missing={asset.source}")
        return None
    target = output_dir / asset.target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset.source, target)
    return target


def manifest_rows(output_dir: Path) -> list[str]:
    rows: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "LAUNCH_KIT_MANIFEST.md":
            rows.append(
                f"| `{path.relative_to(output_dir).as_posix()}` | {path.stat().st_size:,} | `{file_sha256(path)}` |"
            )
    return rows


def write_manifest(output_dir: Path, version: str, release_url: str, generated_date: str) -> Path:
    lines = [
        "# GPT2-BASIC Launch Kit",
        "",
        f"Version: `{version}`",
        f"Release: {release_url}",
        f"Generated: `{generated_date}`",
        "",
        "This handoff bundle contains the verified preview release zips, DOSBox bundle, hardware-transfer zip, launch media, thumbnail, and reusable launch copy.",
        "",
        "## Recommended Use",
        "",
        "- Upload or share the MP4 files as launch clips.",
        "- Use `thumbnail_gpt_in_dos.png` as the default video thumbnail.",
        "- Use `copy/promo-kit.md` for social posts, descriptions, and press copy.",
        "- Use `release/gpt2-basic-preview.zip` as the main downloadable preview payload.",
        "- Use `release/gpt2-basic-dosbox.zip` for a ready-to-mount DOSBox demo.",
        "- Use `release/gpt2-basic-hardware-transfer.zip` for DOS-machine transfer tests.",
        "",
        "## Assets",
        "",
        "| Path | Bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    lines.extend(manifest_rows(output_dir))
    lines.append("")
    manifest = output_dir / "LAUNCH_KIT_MANIFEST.md"
    manifest.write_text("\n".join(lines), encoding="ascii")
    return manifest


def create_zip(output_dir: Path, zip_path: Path, force: bool) -> None:
    if zip_path.exists():
        if not force:
            raise SystemExit(f"LAUNCH_KIT_FAILED zip_exists={zip_path} use --force")
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                write_deterministic_zip_entry(archive, output_dir.parent, path)


def build_launch_kit(
    output_dir: Path,
    zip_path: Path,
    zip_sha256: Path,
    force: bool,
    version: str,
    release_url: str,
    generated_date: str,
) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(f"LAUNCH_KIT_FAILED output_exists={output_dir} use --force")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    verify_release_sidecars()
    seen_targets: set[str] = set()
    for asset in all_assets():
        require(asset.target not in seen_targets, f"duplicate_target={asset.target}")
        seen_targets.add(asset.target)
        copy_asset(asset, output_dir)
    write_manifest(output_dir, version, release_url, generated_date)
    create_zip(output_dir, zip_path, force)
    write_zip_checksum(zip_path, zip_sha256)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.txt"
        source.write_text("hello\n", encoding="ascii")
        output = root / "kit"
        output.mkdir()
        payload = root / "payload.zip"
        payload.write_text("payload\n", encoding="ascii")
        payload_sha = root / "payload.zip.sha256"
        write_zip_checksum(payload, payload_sha)
        verify_zip_sidecar(payload, payload_sha)
        copied = copy_asset(KitAsset(source, "copy/source.txt"), output)
        require(copied is not None and copied.read_text(encoding="ascii") == "hello\n", "copy_asset_self_test")
        write_manifest(output, "test", "https://example.invalid/release", "2026-01-01")
        zip_path = root / "kit.zip"
        create_zip(output, zip_path, False)
        sha_path = root / "kit.zip.sha256"
        write_zip_checksum(zip_path, sha_path)
        first_checksum = sha_path.read_text(encoding="ascii")
        zip_path.unlink()
        create_zip(output, zip_path, False)
        write_zip_checksum(zip_path, sha_path)
        require(first_checksum == sha_path.read_text(encoding="ascii"), "deterministic_zip_self_test")
        require(zip_path.exists() and sha_path.exists(), "zip_self_test")
    print("PROBE_OK launch_kit_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--zip-sha256", type=Path, default=DEFAULT_ZIP_SHA256)
    parser.add_argument("--version", default=DEFAULT_RELEASE_VERSION)
    parser.add_argument("--release-url", default=DEFAULT_RELEASE_URL)
    parser.add_argument("--generated-date", default=DEFAULT_GENERATED_DATE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    build_launch_kit(
        args.output_dir,
        args.zip_path,
        args.zip_sha256,
        args.force,
        args.version,
        args.release_url,
        args.generated_date,
    )
    print(f"LAUNCH_KIT_TREE|path={args.output_dir}")
    print(f"LAUNCH_KIT_ZIP|path={args.zip_path}")
    print(f"LAUNCH_KIT_ZIP_SHA256|path={args.zip_sha256}")


if __name__ == "__main__":
    main()

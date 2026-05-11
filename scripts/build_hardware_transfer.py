#!/usr/bin/env python3
"""Build the minimal DOS transfer tree for physical hardware validation."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/private/tmp/gpt2-basic-hardware-transfer")
DEFAULT_ZIP = Path("/private/tmp/gpt2-basic-hardware-transfer.zip")
DEFAULT_ZIP_SHA256 = Path("/private/tmp/gpt2-basic-hardware-transfer.zip.sha256")
DEFAULT_MODEL_DIR = ROOT / "assets" / "gpt2_basic" / "MODEL"
DEFAULT_PACK_DIR = ROOT / "assets" / "gpt2_basic" / "PACKS"
DEFAULT_GPT2_EXE = ROOT / "qemu" / "evidence" / "GPT2.EXE"
DEFAULT_STAGED_SRC = ROOT / "qemu" / "staging" / "GPT2SRC"
DETERMINISTIC_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"HARDWARE_TRANSFER_FAILED {message}")


def short_name_ok(path: Path) -> bool:
    name = path.name
    base, dot, ext = name.partition(".")
    if not base or len(base) > 8 or len(ext) > 3 or "." in ext:
        return False
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$~!#%&-{}()@'`"
    return all(ch in allowed for ch in (base + ext).upper())


def validate_83(root: Path) -> None:
    for path in root.rglob("*"):
        if not short_name_ok(path):
            raise SystemExit(f"HARDWARE_TRANSFER_FAILED non_8_3={path.relative_to(root)}")


def copy_tree(src: Path, dst: Path) -> None:
    require(src.is_dir(), f"missing_dir={src}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"))


def copy_file(src: Path, dst: Path) -> None:
    require(src.is_file(), f"missing_file={src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_lines(output_dir: Path) -> list[str]:
    lines = [
        "GPT2-BASIC physical hardware transfer manifest",
        "Copy the GPT2 directory to C:\\GPT2 on the target DOS machine.",
        "Run C:, CD \\GPT2, then HWVALID.BAT.",
        "",
        "SHA256  BYTES  PATH",
    ]
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(output_dir).as_posix()
        lines.append(f"{sha256(path)}  {path.stat().st_size}  {rel}")
    return lines


def write_zip(output_dir: Path, zip_path: Path, force: bool) -> None:
    if zip_path.exists():
        require(force, f"zip_exists={zip_path}")
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(output_dir.parent).as_posix(), date_time=DETERMINISTIC_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())


def write_zip_checksum(zip_path: Path, checksum_path: Path) -> None:
    checksum_path.write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="ascii")


def build_transfer(
    output_dir: Path,
    zip_path: Path,
    zip_sha256: Path,
    model_dir: Path,
    pack_dir: Path,
    gpt2_exe: Path,
    staged_src: Path,
    force: bool,
    refresh_staging: bool,
) -> None:
    if refresh_staging:
        subprocess.run([sys.executable, str(ROOT / "qemu" / "make_dos_staging.py")], check=True)

    if output_dir.exists():
        require(force, f"output_exists={output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    dos_root = output_dir / "GPT2"
    dos_root.mkdir()
    copy_file(gpt2_exe, dos_root / "GPT2.EXE")
    copy_file(ROOT / "hardware" / "HWVALID.BAT", dos_root / "HWVALID.BAT")
    copy_file(ROOT / "hardware" / "HWNOTES.TXT", dos_root / "HWNOTES.TXT")
    copy_tree(model_dir, dos_root / "MODEL")
    copy_tree(pack_dir, dos_root / "PACKS")
    copy_tree(staged_src, dos_root / "GPT2SRC")

    validate_83(dos_root)
    (output_dir / "README.TXT").write_text(
        "GPT2-BASIC hardware transfer bundle.\n"
        "Copy GPT2 to C:\\GPT2 on the DOS target, fill HWNOTES.TXT, then run HWVALID.BAT.\n",
        encoding="ascii",
    )
    (output_dir / "MANIFEST.TXT").write_text("\n".join(manifest_lines(output_dir)) + "\n", encoding="ascii")
    write_zip(output_dir, zip_path, force)
    write_zip_checksum(zip_path, zip_sha256)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "MODEL"
        packs = root / "PACKS"
        staged = root / "GPT2SRC"
        model.mkdir()
        packs.mkdir()
        staged.mkdir()
        (root / "GPT2.EXE").write_bytes(b"exe")
        for name in ("GPT2CFG.TXT", "GPT2FX.BIN", "GPT2EXP.BIN"):
            (model / name).write_bytes(b"model")
        (packs / "PACKS.TXT").write_text("DOSHELP\n", encoding="ascii")
        (staged / "ASSIST.BAS").write_text("PRINT \"ASSIST\"\n", encoding="ascii")
        out = root / "OUT"
        zip_path = root / "OUT.ZIP"
        zip_sha256 = root / "OUT.SHA"
        build_transfer(out, zip_path, zip_sha256, model, packs, root / "GPT2.EXE", staged, force=True, refresh_staging=False)
        require((out / "GPT2" / "HWVALID.BAT").exists(), "self_test_missing_hwvalid")
        require((out / "README.TXT").exists(), "self_test_missing_readme")
        require((out / "MANIFEST.TXT").exists(), "self_test_missing_manifest")
        require(zip_path.exists(), "self_test_missing_zip")
        require(zip_sha256.exists(), "self_test_missing_zip_sha256")
        validate_83(out / "GPT2")
    print("PROBE_OK hardware_transfer_8_3=1")
    print("PROBE_OK hardware_transfer_readme=README.TXT")
    print("PROBE_OK hardware_transfer_manifest=1")
    print("PROBE_OK hardware_transfer_zip=1")
    print("PROBE_OK hardware_transfer_zip_sha256=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--zip-sha256", type=Path, default=DEFAULT_ZIP_SHA256)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--gpt2-exe", type=Path, default=DEFAULT_GPT2_EXE)
    parser.add_argument("--staged-src", type=Path, default=DEFAULT_STAGED_SRC)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-refresh-staging", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    build_transfer(
        args.output_dir,
        args.zip_path,
        args.zip_sha256,
        args.model_dir,
        args.pack_dir,
        args.gpt2_exe,
        args.staged_src,
        force=args.force,
        refresh_staging=not args.no_refresh_staging,
    )
    print(f"HARDWARE_TRANSFER_TREE|path={args.output_dir}")
    print(f"HARDWARE_TRANSFER_ZIP|path={args.zip_path}")
    print(f"HARDWARE_TRANSFER_ZIP_SHA256|path={args.zip_sha256}")
    print("PROBE_OK hardware_transfer=1")


if __name__ == "__main__":
    main()

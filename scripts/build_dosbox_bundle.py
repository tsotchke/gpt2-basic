#!/usr/bin/env python3
"""Build a DOSBox-ready GPT2-BASIC bundle."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

try:
    import build_hardware_transfer
except ModuleNotFoundError:  # pragma: no cover - import path used by unittest package imports.
    from scripts import build_hardware_transfer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/private/tmp/gpt2-basic-dosbox")
DEFAULT_ZIP = Path("/private/tmp/gpt2-basic-dosbox.zip")
DEFAULT_ZIP_SHA256 = Path("/private/tmp/gpt2-basic-dosbox.zip.sha256")
DEFAULT_MODEL_DIR = ROOT / "assets" / "gpt2_basic" / "MODEL"
DEFAULT_PACK_DIR = ROOT / "assets" / "gpt2_basic" / "PACKS"
DEFAULT_GPT2_EXE = ROOT / "qemu" / "evidence" / "GPT2.EXE"
DEFAULT_ASSIST_EXE = ROOT / "qemu" / "evidence" / "ASSIST.EXE"
DEFAULT_STAGED_SRC = ROOT / "qemu" / "staging" / "GPT2SRC"
DEFAULT_DPMI_HOST = ROOT / "third_party" / "cwsdpmi" / "CWSDPMI.EXE"
DEFAULT_DPMI_LICENSE = ROOT / "third_party" / "cwsdpmi" / "COPYING.CWS"
DEFAULT_DPMI_LSM = ROOT / "third_party" / "cwsdpmi" / "CWSDPMI.LSM"
DETERMINISTIC_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class DosboxProfile:
    name: str
    conf_name: str
    description: str
    commands: tuple[str, ...]


DOSBOX_PROFILES = (
    DosboxProfile(
        "manual",
        "GPT2MAN.CONF",
        "Mount the bundle and leave DOSBox at C:\\GPT2.",
        (
            "echo GPT2-BASIC mounted at C:\\GPT2",
            "echo Try: ASSIST.EXE",
            "echo Try: GPT2.EXE",
            "echo Try: GPT2.EXE --demo",
            "echo Try: GPT2.EXE --quality-all",
            "echo Try: GPT2.EXE --perf",
            "dir",
        ),
    ),
    DosboxProfile(
        "chat",
        "GPT2CHAT.CONF",
        "Run the pack-driven GPT2-BASIC conversational assistant.",
        (
            "echo GPT2-BASIC assistant shell",
            "echo Commands: /packs, /pack CHAT, /pack DOSHELP, /pack OFFICE, /about, /quit",
            "echo.",
            "ASSIST.EXE",
            "echo.",
            "echo Assistant exited. You are back at C:\\GPT2.",
        ),
    ),
    DosboxProfile(
        "completion",
        "GPT2INT.CONF",
        "Run GPT2-BASIC prompt-based interactive completion.",
        (
            "echo GPT2-BASIC interactive completion",
            "echo Enter a prompt, finish with a blank line, then wait for output.",
            "echo.",
            "GPT2.EXE",
            "echo.",
            "echo Interactive session exited. You are back at C:\\GPT2.",
        ),
    ),
    DosboxProfile(
        "demo",
        "GPT2DEMO.CONF",
        "Run the GPT2-BASIC demo and leave the DOSBox shell open.",
        (
            "GPT2.EXE --demo",
            "echo.",
            "echo Demo complete. Output stayed on screen.",
        ),
    ),
    DosboxProfile(
        "quality",
        "GPT2QUAL.CONF",
        "Run the all-suite quality probe and write C:\\GPT2\\QUAL.LOG.",
        (
            "GPT2.EXE --quality-all > QUAL.LOG",
            "type QUAL.LOG",
        ),
    ),
    DosboxProfile(
        "perf",
        "GPT2PERF.CONF",
        "Run the performance probe and write C:\\GPT2\\PERF.LOG.",
        (
            "GPT2.EXE --perf > PERF.LOG",
            "type PERF.LOG",
        ),
    ),
    DosboxProfile(
        "trace",
        "GPT2TRAC.CONF",
        "Run the educational trace and write C:\\GPT2\\TRACE.LOG.",
        (
            "GPT2.EXE --trace > TRACE.LOG",
            "type TRACE.LOG",
        ),
    ),
    DosboxProfile(
        "sampling",
        "GPT2SAMP.CONF",
        "Run the sampling matrix and write C:\\GPT2\\SAMPLE.LOG.",
        (
            "GPT2.EXE --sampling-matrix > SAMPLE.LOG",
            "type SAMPLE.LOG",
        ),
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"DOSBOX_BUNDLE_FAILED {message}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip_checksum(zip_path: Path, zip_sha256: Path) -> None:
    zip_sha256.parent.mkdir(parents=True, exist_ok=True)
    zip_sha256.write_text(f"{file_sha256(zip_path)}  {zip_path.name}\n", encoding="ascii")


def write_deterministic_zip_entry(archive: zipfile.ZipFile, root_parent: Path, path: Path) -> None:
    info = zipfile.ZipInfo(path.relative_to(root_parent).as_posix(), date_time=DETERMINISTIC_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, path.read_bytes())


def create_zip(output_dir: Path, zip_path: Path, force: bool) -> None:
    if zip_path.exists():
        require(force, f"zip_exists={zip_path} use --force")
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                write_deterministic_zip_entry(archive, output_dir.parent, path)


def profile_conf(profile: DosboxProfile) -> str:
    lines = [
        "# GPT2-BASIC DOSBox profile",
        f"# Profile: {profile.name}",
        "# Run from the gpt2-basic-dosbox directory with:",
        f"#   dosbox -conf DOSBOX/{profile.conf_name}",
        "",
        "[sdl]",
        "fullscreen=false",
        "autolock=false",
        "waitonerror=true",
        "",
        "[dosbox]",
        "machine=svga_s3",
        "memsize=64",
        "",
        "[render]",
        "aspect=true",
        "scaler=normal2x",
        "",
        "[cpu]",
        "core=dynamic",
        "cycles=max",
        "",
        "[dos]",
        "xms=true",
        "ems=true",
        "umb=true",
        "",
        "[autoexec]",
        "@echo off",
        "mount c .",
        "c:",
        "cd \\GPT2",
    ]
    lines.extend(profile.commands)
    lines.append("")
    return "\n".join(lines)


def shell_launcher(profile: DosboxProfile) -> str:
    return (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        "cd \"$(dirname \"$0\")\"\n"
        f"\"${{DOSBOX:-dosbox}}\" -conf DOSBOX/{profile.conf_name}\n"
    )


def windows_launcher(profile: DosboxProfile) -> str:
    return (
        "@echo off\r\n"
        f"dosbox -conf DOSBOX\\{profile.conf_name}\r\n"
    )


def write_readme(output_dir: Path) -> None:
    lines = [
        "GPT2-BASIC DOSBox bundle",
        "",
        "Run from this directory after installing DOSBox, DOSBox Staging, or DOSBox-X.",
        "The generated configs mount this directory as C: and start in C:\\GPT2.",
        "",
        "Recommended commands:",
        "",
        "  dosbox -conf DOSBOX/GPT2MAN.CONF",
        "  dosbox -conf DOSBOX/GPT2CHAT.CONF",
        "  dosbox -conf DOSBOX/GPT2INT.CONF",
        "  dosbox -conf DOSBOX/GPT2DEMO.CONF",
        "  dosbox -conf DOSBOX/GPT2QUAL.CONF",
        "  dosbox -conf DOSBOX/GPT2PERF.CONF",
        "",
        "On Unix-like hosts you can also run:",
        "",
        "  ./run-chat.sh",
        "  ./run-demo.sh",
        "  ./run-completion.sh",
        "  ./run-quality.sh",
        "  ./run-perf.sh",
        "",
        "On Windows, use the matching RUN*.BAT launchers.",
        "",
        "Profiles:",
    ]
    for profile in DOSBOX_PROFILES:
        lines.append(f"  {profile.conf_name}: {profile.description}")
    lines.extend(
        [
            "",
            "Logs written by DOSBox runs stay in GPT2, for example QUAL.LOG, PERF.LOG, TRACE.LOG, and SAMPLE.LOG.",
            "This is a convenience emulator path, not a replacement for the QEMU release gate or physical 486 validation.",
            "",
        ]
    )
    (output_dir / "README.TXT").write_text("\n".join(lines), encoding="ascii")


def write_manifest(output_dir: Path) -> None:
    lines = [
        "GPT2-BASIC DOSBox bundle manifest",
        "SHA256  BYTES  PATH",
    ]
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.TXT":
            rel = path.relative_to(output_dir).as_posix()
            lines.append(f"{file_sha256(path)}  {path.stat().st_size}  {rel}")
    (output_dir / "MANIFEST.TXT").write_text("\n".join(lines) + "\n", encoding="ascii")


def write_dosbox_files(output_dir: Path) -> None:
    conf_dir = output_dir / "DOSBOX"
    conf_dir.mkdir(parents=True, exist_ok=True)
    for profile in DOSBOX_PROFILES:
        (conf_dir / profile.conf_name).write_text(profile_conf(profile), encoding="ascii")
        (output_dir / f"run-{profile.name}.sh").write_text(shell_launcher(profile), encoding="ascii")
        (output_dir / f"RUN{profile.name[:4].upper()}.BAT").write_text(windows_launcher(profile), encoding="ascii")
    write_readme(output_dir)
    write_manifest(output_dir)


def build_dosbox_bundle(
    output_dir: Path,
    zip_path: Path,
    zip_sha256: Path,
    model_dir: Path,
    pack_dir: Path,
    gpt2_exe: Path,
    assist_exe: Path | None,
    staged_src: Path,
    force: bool,
    refresh_staging: bool,
    dpmi_host: Path | None = DEFAULT_DPMI_HOST,
    dpmi_license: Path | None = DEFAULT_DPMI_LICENSE,
    dpmi_lsm: Path | None = DEFAULT_DPMI_LSM,
) -> None:
    if output_dir.exists():
        require(force, f"output_exists={output_dir} use --force")
        shutil.rmtree(output_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        transfer_zip = tmp_root / "transfer.zip"
        transfer_sha = tmp_root / "transfer.zip.sha256"
        build_hardware_transfer.build_transfer(
            output_dir,
            transfer_zip,
            transfer_sha,
            model_dir,
            pack_dir,
            gpt2_exe,
            staged_src,
            force=True,
            refresh_staging=refresh_staging,
            assist_exe=assist_exe,
            dpmi_host=dpmi_host,
            dpmi_license=dpmi_license,
            dpmi_lsm=dpmi_lsm,
        )

    write_dosbox_files(output_dir)
    create_zip(output_dir, zip_path, force)
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
        (root / "ASSIST.EXE").write_bytes(b"assistant")
        (root / "CWSDPMI.EXE").write_bytes(b"dpmi")
        (root / "COPYING.CWS").write_text("license\n", encoding="ascii")
        (root / "CWSDPMI.LSM").write_text("source\n", encoding="ascii")
        for name in ("GPT2CFG.TXT", "GPT2FX.BIN", "GPT2EXP.BIN", "VOCAB.BIN"):
            (model / name).write_bytes(b"model")
        (packs / "PACKS.TXT").write_text("CHAT\n", encoding="ascii")
        (staged / "MAIN.BAS").write_text("PRINT \"MAIN\"\n", encoding="ascii")
        out = root / "dosbox"
        zip_path = root / "dosbox.zip"
        zip_sha = root / "dosbox.zip.sha256"
        build_dosbox_bundle(
            out,
            zip_path,
            zip_sha,
            model,
            packs,
            root / "GPT2.EXE",
            root / "ASSIST.EXE",
            staged,
            True,
            False,
            dpmi_host=root / "CWSDPMI.EXE",
            dpmi_license=root / "COPYING.CWS",
            dpmi_lsm=root / "CWSDPMI.LSM",
        )

        manual_conf = (out / "DOSBOX" / "GPT2MAN.CONF").read_text(encoding="ascii")
        require("mount c ." in manual_conf, "self_test_missing_relative_mount")
        require("cd \\GPT2" in manual_conf, "self_test_missing_gpt2_cd")
        require(str(out) not in manual_conf, "self_test_conf_leaks_host_path")
        require((out / "GPT2" / "CWSDPMI.EXE").exists(), "self_test_missing_dpmi")
        require((out / "GPT2" / "ASSIST.EXE").exists(), "self_test_missing_assist")
        require((out / "DOSBOX" / "GPT2CHAT.CONF").exists(), "self_test_missing_chat_conf")
        require("ASSIST.EXE" in (out / "DOSBOX" / "GPT2CHAT.CONF").read_text(encoding="ascii"), "self_test_chat_conf")
        require((out / "DOSBOX" / "GPT2INT.CONF").exists(), "self_test_missing_interactive_conf")
        require("GPT2.EXE" in (out / "DOSBOX" / "GPT2INT.CONF").read_text(encoding="ascii"), "self_test_interactive_conf")
        require((out / "run-demo.sh").exists(), "self_test_missing_shell_launcher")
        require((out / "RUNDEMO.BAT").exists(), "self_test_missing_windows_launcher")
        require((out / "MANIFEST.TXT").exists(), "self_test_missing_manifest")
        require(zip_path.exists() and zip_sha.exists(), "self_test_missing_zip")

    print("PROBE_OK dosbox_bundle_profiles=8")
    print("PROBE_OK dosbox_bundle_relative_mount=1")
    print("PROBE_OK dosbox_bundle_dpmi_host=1")
    print("PROBE_OK dosbox_bundle_assistant_shell=1")
    print("PROBE_OK dosbox_bundle_interactive_profile=1")
    print("PROBE_OK dosbox_bundle_launchers=1")
    print("PROBE_OK dosbox_bundle_zip=1")
    print("PROBE_OK dosbox_bundle_zip_sha256=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--zip-sha256", type=Path, default=DEFAULT_ZIP_SHA256)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--gpt2-exe", type=Path, default=DEFAULT_GPT2_EXE)
    parser.add_argument("--assist-exe", type=Path, default=DEFAULT_ASSIST_EXE)
    parser.add_argument("--staged-src", type=Path, default=DEFAULT_STAGED_SRC)
    parser.add_argument("--no-assist-exe", action="store_true")
    parser.add_argument("--dpmi-host", type=Path, default=DEFAULT_DPMI_HOST)
    parser.add_argument("--dpmi-license", type=Path, default=DEFAULT_DPMI_LICENSE)
    parser.add_argument("--dpmi-lsm", type=Path, default=DEFAULT_DPMI_LSM)
    parser.add_argument("--no-dpmi-host", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-refresh-staging", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    build_dosbox_bundle(
        args.output_dir,
        args.zip_path,
        args.zip_sha256,
        args.model_dir,
        args.pack_dir,
        args.gpt2_exe,
        None if args.no_assist_exe else args.assist_exe,
        args.staged_src,
        args.force,
        not args.no_refresh_staging,
        dpmi_host=None if args.no_dpmi_host else args.dpmi_host,
        dpmi_license=None if args.no_dpmi_host else args.dpmi_license,
        dpmi_lsm=None if args.no_dpmi_host else args.dpmi_lsm,
    )
    print(f"DOSBOX_BUNDLE_TREE|path={args.output_dir}")
    print(f"DOSBOX_BUNDLE_ZIP|path={args.zip_path}")
    print(f"DOSBOX_BUNDLE_ZIP_SHA256|path={args.zip_sha256}")
    print("PROBE_OK dosbox_bundle=1")


if __name__ == "__main__":
    main()

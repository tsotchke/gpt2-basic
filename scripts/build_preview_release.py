#!/usr/bin/env python3
"""Build a bounded GPT2-BASIC preview-release tree.

The preview release is intentionally narrower than the repository. It includes
only strict-quality release models, assistant packs, source/rebuild scripts, and
selected evidence. Historical rejected candidates stay in the repo as training
evidence, but they are not shipped as latent release options.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from audit_exported_models import (
    DEFAULT_EVIDENCE,
    DEFAULT_MODELS_ROOT,
    DEFAULT_PACK_ROOT,
    AuditRow,
    audit_models,
    required_quality,
    shape_text,
)
from plan_model_quality_repairs import RELEASE_MODEL_NAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/private/tmp/gpt2-basic-preview")
DEFAULT_ZIP = Path("/private/tmp/gpt2-basic-preview.zip")
DEFAULT_ZIP_SHA256 = Path("/private/tmp/gpt2-basic-preview.zip.sha256")
DEFAULT_MANIFEST = DEFAULT_EVIDENCE / "preview_release_manifest.md"
DEFAULT_VERSION = "v0.1.0-preview"
DEFAULT_GENERATED_DATE = "2026-05-12"
DETERMINISTIC_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ReleaseModel:
    name: str
    label: str
    reason: str


RELEASE_MODELS = (
    ReleaseModel("MODEL", "default full resident", "current slim production default"),
    ReleaseModel("MODEL_LEXICON_GOLD_V4_S3000", "gold-v4 full resident", "best measured full-resident quality checkpoint"),
    ReleaseModel("MODEL_HEADSHORTLIST2048_PROD_PROBE", "head shortlist", "fastest measured large-vocabulary path"),
    ReleaseModel("MODEL_HEADQ4_PROD_PROBE", "q4 output head", "compressed output-head candidate"),
    ReleaseModel("MODEL_TOKHEADQ4_PROD_PROBE", "q4 token+head", "best current low-memory default"),
    ReleaseModel("MODEL_TOKHEADQ4_STREAM_PROD_PROBE", "q4 streamed head", "maximum RAM compatibility fallback"),
)

DOC_FILES = (
    "README.md",
    "qemu/README.md",
    "gpt2_basic_tldr.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "requirements-promo.txt",
    "docs/dosbox.md",
    "docs/hardware-validation.md",
    "docs/public-launch.md",
    "docs/marketing/promo-kit.md",
    "docs/marketing/public-demo-script.md",
    "docs/marketing/video-plan.md",
    "docs/releases/v0.1.0-preview.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/hardware_validation.yml",
    ".github/workflows/preview-release.yml",
)
TREE_DIRS = ("src", "scripts", "tests", "data/domain_curriculum", "hardware")
QEMU_FILE_SUFFIXES = (".sh", ".bat", ".py")
SELECTED_EVIDENCE_NAMES = {
    "GPT2.EXE",
    "assistant_486.log",
    "assistant_chat_manual_probe_2026-05-12.md",
    "assistant_compile_486.log",
    "assistant_consistency_eval.md",
    "assistant_generalist_repair.md",
    "assistant_generalist_prompt_eval.md",
    "assistant_interactive_chat_486.md",
    "assistant_pack_retrieval_eval.md",
    "assistant_pack_probe.log",
    "assistant_raw_prompt_eval.md",
    "assistant_stress_486.log",
    "assistant_stress_compile_486.log",
    "assistant_stress_report.md",
    "compile_main_486.log",
    "exported_model_quality_inventory.md",
    "gold_curriculum_v5_clean_repair_report.md",
    "hardware_capture_486_qemu_probe.log",
    "hardware_capture_probe.log",
    "hardware_performance_matrix.md",
    "hardware_perf_report.md",
    "hardware_transfer_probe.log",
    "improvement_backlog.md",
    "model_quality_repair_plan.md",
    "perf_486_486dx2-66_model_lexicon_gold_v4_s3000.log",
    "preview_release_probe.log",
    "quality_486_model_lexicon_gold_v4_s3000.log",
    "quality_report_dos.md",
    "quality_report_dos_all.md",
    "quality_report_dos_heldout.md",
    "run_main_486.log",
    "vector_486_model_lexicon_gold_v4_s3000.log",
    "workspace_tracking_probe.log",
}
SELECTED_EVIDENCE_PREFIXES = (
    "quality_report_default_model",
    "quality_report_dos_model_lexicon_gold_v4_s3000",
    "quality_report_headshortlist2048",
    "quality_report_headq4",
    "quality_report_tokheadq4",
    "quality_report_lexicon_gold_v4_s3000",
    "quality_report_assistant_",
)
SELECTED_EVIDENCE_DIRS = {
    "hardware_capture_486_qemu",
}
TREE_COPY_IGNORED_NAMES = {".DS_Store", "__pycache__"}
TREE_COPY_IGNORED_SUFFIXES = {".pyc", ".pyo"}
SANITIZED_TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".tsv",
    ".txt",
    ".yml",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def dir_stats(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_package_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / "SHA256SUMS.txt"
    lines: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        lines.append(f"{file_sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return checksum_path


def write_zip_checksum(zip_path: Path, zip_sha256: Path) -> None:
    zip_sha256.parent.mkdir(parents=True, exist_ok=True)
    zip_sha256.write_text(f"{file_sha256(zip_path)}  {zip_path.name}\n", encoding="ascii")


def write_deterministic_zip_entry(archive: zipfile.ZipFile, root_parent: Path, path: Path) -> None:
    info = zipfile.ZipInfo(path.relative_to(root_parent).as_posix(), date_time=DETERMINISTIC_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, path.read_bytes())


def quality_text(row: AuditRow) -> str:
    report = required_quality(row.quality_reports)
    if report is None:
        return "missing"
    return f"{report.status} {report.passed}/{report.total} avg {report.average:.3f} ({report.backend}, {report.suite})"


def release_rows(rows: list[AuditRow]) -> list[tuple[ReleaseModel, AuditRow]]:
    by_name = {row.model.name: row for row in rows}
    selected: list[tuple[ReleaseModel, AuditRow]] = []
    missing = [model.name for model in RELEASE_MODELS if model.name not in by_name]
    if missing:
        raise SystemExit("PREVIEW_RELEASE_FAILED missing_models=" + ",".join(missing))

    for model in RELEASE_MODELS:
        row = by_name[model.name]
        report = required_quality(row.quality_reports)
        if not row.artifact_ok:
            raise SystemExit(f"PREVIEW_RELEASE_FAILED artifact={model.name}")
        if report is None or report.status != "PASS" or report.passed != report.total:
            raise SystemExit(f"PREVIEW_RELEASE_FAILED quality={model.name}")
        selected.append((model, row))
    return selected


def assistant_rows(rows: list[AuditRow]) -> list[AuditRow]:
    selected = [row for row in rows if row.model.role == "assistant_pack"]
    raw_prompt_report_path = DEFAULT_EVIDENCE / "assistant_raw_prompt_eval.md"
    raw_prompt_report = raw_prompt_report_path.read_text(encoding="ascii", errors="ignore") if raw_prompt_report_path.exists() else ""
    raw_prompt_match = re.search(r"Prompt pass rate:\s+`(\d+)/(\d+)`", raw_prompt_report)
    raw_prompt_passed = False
    if raw_prompt_match is not None:
        raw_passed = int(raw_prompt_match.group(1))
        raw_total = int(raw_prompt_match.group(2))
        raw_prompt_passed = "Status: `PASS`" in raw_prompt_report and raw_total >= 67 and raw_passed == raw_total
    for row in selected:
        report = required_quality(row.quality_reports)
        if not row.artifact_ok:
            raise SystemExit(f"PREVIEW_RELEASE_FAILED assistant_artifact={row.model.name}")
        if row.model.name == "ASSISTANT_CHAT" and raw_prompt_passed:
            if report is not None and report.total > 0 and report.passed / report.total >= 0.75:
                continue
        if report is None or report.status != "PASS" or report.passed != report.total:
            raise SystemExit(f"PREVIEW_RELEASE_FAILED assistant_quality={row.model.name}")
    return selected


def selected_evidence(evidence_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(evidence_dir.iterdir()):
        if path.is_dir() and path.name in SELECTED_EVIDENCE_DIRS:
            files.extend(sorted(child for child in path.iterdir() if child.is_file()))
            continue
        if not path.is_file():
            continue
        if path.name in SELECTED_EVIDENCE_NAMES or path.name.startswith(SELECTED_EVIDENCE_PREFIXES):
            files.append(path)
    return files


def tree_copy_skips(path: Path) -> bool:
    if path.name in TREE_COPY_IGNORED_NAMES or path.suffix in TREE_COPY_IGNORED_SUFFIXES:
        return True
    return any(part in TREE_COPY_IGNORED_NAMES for part in path.parts)


def sanitize_public_text(text: str) -> str:
    root_text = ROOT.as_posix()
    resolved_text = ROOT.resolve().as_posix()
    data_volume_text = "/System/Volumes/Data" + resolved_text
    for local_path in (data_volume_text, resolved_text, root_text):
        text = text.replace(local_path, "<repo>")
    return text


def should_sanitize_text(path: Path) -> bool:
    return path.suffix.lower() in SANITIZED_TEXT_SUFFIXES


def copied_tree_files(src: Path) -> list[Path]:
    if not src.exists():
        return []
    return sorted(path for path in src.rglob("*") if path.is_file() and not tree_copy_skips(path))


def release_source_files(
    selected: list[tuple[ReleaseModel, AuditRow]],
    evidence_files: list[Path],
    pack_root: Path,
) -> list[Path]:
    files: list[Path] = []
    files.extend(ROOT / doc for doc in DOC_FILES if (ROOT / doc).is_file())
    for tree in TREE_DIRS:
        files.extend(copied_tree_files(ROOT / tree))
    files.extend(path for path in sorted((ROOT / "qemu").iterdir()) if path.is_file() and path.suffix in QEMU_FILE_SUFFIXES)
    qemu_readme = ROOT / "qemu" / "README.md"
    if qemu_readme.is_file():
        files.append(qemu_readme)
    for _model, row in selected:
        files.extend(copied_tree_files(row.model.path))
    files.extend(copied_tree_files(pack_root))
    gpt2_exe = DEFAULT_EVIDENCE / "GPT2.EXE"
    if gpt2_exe.is_file():
        files.append(gpt2_exe)
    files.extend(path for path in evidence_files if path.is_file())

    unique: dict[str, Path] = {}
    for path in files:
        unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def git_tracked_paths(root: Path = ROOT) -> set[str] | None:
    if not (root / ".git").exists():
        return None
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return {entry for entry in result.stdout.decode("utf-8").split("\0") if entry}


def untracked_release_inputs(source_files: list[Path], tracked_paths: set[str], root: Path = ROOT) -> list[str]:
    missing: list[str] = []
    root_resolved = root.resolve()
    for path in source_files:
        try:
            rel_path = path.resolve().relative_to(root_resolved).as_posix()
        except ValueError:
            continue
        if rel_path not in tracked_paths:
            missing.append(rel_path)
    return sorted(missing)


def require_release_sources_tracked(source_files: list[Path]) -> None:
    tracked = git_tracked_paths()
    if tracked is None:
        return
    missing = untracked_release_inputs(source_files, tracked)
    if missing:
        sample = ",".join(missing[:10])
        raise SystemExit(f"PREVIEW_RELEASE_FAILED untracked_release_input={sample}")


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"))


def copy_if_exists(src: Path, dst: Path, sanitize_text: bool = False) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sanitize_text and should_sanitize_text(src):
        text = src.read_text(encoding="ascii", errors="ignore")
        dst.write_text(sanitize_public_text(text), encoding="ascii")
    else:
        shutil.copy2(src, dst)
    return True


def write_manifest(
    rows: list[AuditRow],
    selected: list[tuple[ReleaseModel, AuditRow]],
    assistants: list[AuditRow],
    evidence_files: list[Path],
    output_dir: Path,
    zip_path: Path,
    package_built: bool,
    generated_date: str,
) -> str:
    included_names = {model.name for model, _row in selected}
    excluded_release_names = sorted(RELEASE_MODEL_NAMES - included_names)
    if package_built and output_dir.exists():
        files, bytes_total = dir_stats(output_dir)
        package_size = f"{files} files, {bytes_total:,} bytes"
    elif package_built:
        package_size = "pending build"
    else:
        package_size = "manifest-only"

    lines = [
        "# GPT2-BASIC Preview Release Manifest",
        "",
        f"Version: `{DEFAULT_VERSION}`",
        f"Generated: `{generated_date}`",
        f"Package tree: `{output_dir.name}`",
        f"Package zip: `{zip_path.name}`",
        f"Package checksums: `SHA256SUMS.txt`; zip sidecar: `{zip_path.name}.sha256`",
        f"Package status: `{package_size}`",
        "",
        "This is an iterative preview payload. It ships only strict-quality release models and assistant packs; rejected repair attempts and old candidates remain repo evidence only.",
        "",
        "## Release Models",
        "",
        "| Model | Label | Shape | Quality | Size | Why shipped |",
        "|---|---|---|---|---|---|",
    ]
    for model, row in selected:
        files, bytes_total = dir_stats(row.model.path)
        lines.append(
            f"| `{model.name}` | {model.label} | `{shape_text(row.cfg)}` | "
            f"{quality_text(row)} | {files} files / {bytes_total:,} B | {model.reason} |"
        )

    lines.extend(["", "## Assistant Packs", "", "| Pack Model | Shape | Quality | Size |", "|---|---|---|---|"])
    for row in assistants:
        files, bytes_total = dir_stats(row.model.path.parent)
        lines.append(f"| `{row.model.name}` | `{shape_text(row.cfg)}` | {quality_text(row)} | {files} files / {bytes_total:,} B |")

    lines.extend(
        [
            "",
            "## Included Runtime Surface",
            "",
            "- `bin/GPT2.EXE` when current QEMU evidence includes the compiled DOS binary.",
            "- `assets/gpt2_basic/MODEL*` for the release models listed above.",
            "- `assets/gpt2_basic/PACKS` with CHAT, DOSHELP, OFFICE, and DEV packs, per-pack `USAGE.TXT`, generated `KDB.TXT`, editable `USER.TXT`, pack-local models where available, and sprite/icon slots.",
            "- `src`, `scripts`, `tests`, selected `qemu` helpers, and `data/domain_curriculum` for rebuild and repair iteration.",
            "- `docs/dosbox.md` and `scripts/build_dosbox_bundle.py` for the DOSBox convenience package.",
            "- Selected QEMU and quality evidence under `qemu/evidence`.",
            "",
            "## Deferred Release Scope",
            "",
            "- This preview release is the DOS demo, DOSBox convenience package, and DOS transfer package.",
            "- Windows and OS/2 assistant shells, including the OS/2/Warp package, stay on the later-release track.",
            "",
            "## Explicitly Excluded",
            "",
            "- Historical byte/domain candidates that miss the strict all-suite quality gate.",
            "- Rejected repair attempts such as `MODEL_PROFILE_386_MIN_LEXICON384_REPAIR`.",
            "- Host-only prototypes that are not DOS-ready.",
        ]
    )
    if excluded_release_names:
        lines.append(f"- Release constants not selected by this manifest: `{', '.join(excluded_release_names)}`.")

    lines.extend(
        [
            "",
            "## Physical Hardware Path",
            "",
            "- `docs/hardware-validation.md` defines the non-emulator validation ladder.",
            "- A physical 486-class DOS machine is the solid-release baseline.",
            "- `qemu/run_hardware_capture_486.sh` rehearses the same `C:\\GPT2\\HWVALID.BAT` capture path before physical transfer.",
            "- `scripts/build_hardware_transfer.py` creates the minimal `C:\\GPT2` transfer bundle for the physical machine.",
            "- `C:\\GPT2\\RETURN.TXT` gives the DOS-side copy-back checklist for real machine logs.",
            "- `scripts/stage_hardware_capture_evidence.py` verifies returned physical logs and stages stable release evidence names.",
            "- `scripts/hardware_performance_matrix.py` builds the physical-only performance table from staged logs.",
            "- Pentium hardware is useful for scaling evidence, but it is not a blocker for the 486-focused release.",
            "",
            "## Release Notes",
            "",
            "- `docs/releases/v0.1.0-preview.md` is the GitHub release body for this payload.",
            "- Attach the preview zip, DOSBox zip, hardware-transfer zip, `.sha256` sidecars, and `preview_release_manifest.md` to the GitHub prerelease.",
            "",
            "## Evidence Files",
            "",
        ]
    )
    for path in evidence_files:
        lines.append(f"- `{rel(path)}`")

    lines.extend(
        [
            "",
            "## Verification Commands",
            "",
            "```sh",
            "python3 -m unittest discover tests",
            "python3 scripts/audit_exported_models.py",
            "python3 scripts/verify_assistant_packs.py",
            "python3 scripts/verify_hardware_capture.py --self-test",
            "python3 scripts/stage_hardware_capture_evidence.py --self-test",
            "python3 scripts/hardware_performance_matrix.py --self-test",
            "python3 scripts/build_dosbox_bundle.py --self-test",
            "python3 scripts/build_hardware_transfer.py --self-test",
            "python3 scripts/build_preview_release.py --self-test",
            "python3 scripts/verify_preview_artifacts.py --self-test",
            "python3 scripts/verify_workspace_tracking.py",
            "python3 scripts/build_preview_release.py --force",
            "python3 scripts/build_dosbox_bundle.py --force",
            "python3 scripts/build_hardware_transfer.py --force",
            "python3 scripts/verify_preview_artifacts.py",
            "```",
            "",
            f"Audited models: `{len(rows)}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_package(
    output_dir: Path,
    manifest_text: str,
    selected: list[tuple[ReleaseModel, AuditRow]],
    evidence_files: list[Path],
    pack_root: Path,
    force: bool,
) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(f"PREVIEW_RELEASE_FAILED output_exists={output_dir} use --force")
        shutil.rmtree(output_dir)
    require_release_sources_tracked(release_source_files(selected, evidence_files, pack_root))
    output_dir.mkdir(parents=True)
    for doc in DOC_FILES:
        copy_if_exists(ROOT / doc, output_dir / doc)
    for tree in TREE_DIRS:
        copy_tree(ROOT / tree, output_dir / tree)

    qemu_out = output_dir / "qemu"
    qemu_out.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "qemu").iterdir()):
        if path.is_file() and path.suffix in QEMU_FILE_SUFFIXES:
            copy_if_exists(path, qemu_out / path.name)
    copy_if_exists(ROOT / "qemu" / "README.md", qemu_out / "README.md")

    assets_out = output_dir / "assets" / "gpt2_basic"
    for model, row in selected:
        copy_tree(row.model.path, assets_out / model.name)
    copy_tree(pack_root, assets_out / "PACKS")

    copy_if_exists(DEFAULT_EVIDENCE / "GPT2.EXE", output_dir / "bin" / "GPT2.EXE")
    for path in evidence_files:
        copy_if_exists(path, output_dir / rel(path), sanitize_text=True)

    (output_dir / "preview_release_manifest.md").write_text(manifest_text, encoding="ascii")


def write_converged_package_manifest(
    rows: list[AuditRow],
    selected: list[tuple[ReleaseModel, AuditRow]],
    assistants: list[AuditRow],
    evidence: list[Path],
    output_dir: Path,
    zip_path: Path,
    manifest_path: Path,
    generated_date: str,
) -> str:
    manifest_text = write_manifest(rows, selected, assistants, evidence, output_dir, zip_path, True, generated_date)
    for _ in range(10):
        (output_dir / "preview_release_manifest.md").write_text(manifest_text, encoding="ascii")
        write_package_checksums(output_dir)
        next_text = write_manifest(rows, selected, assistants, evidence, output_dir, zip_path, True, generated_date)
        if next_text == manifest_text:
            manifest_path.write_text(manifest_text, encoding="ascii")
            return manifest_text
        manifest_text = next_text
    raise RuntimeError("preview release manifest status did not converge")


def create_zip(output_dir: Path, zip_path: Path, force: bool) -> None:
    if zip_path.exists():
        if not force:
            raise SystemExit(f"PREVIEW_RELEASE_FAILED zip_exists={zip_path} use --force")
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                write_deterministic_zip_entry(archive, output_dir.parent, path)


def self_test(rows: list[AuditRow], evidence_dir: Path, pack_root: Path) -> None:
    selected = release_rows(rows)
    assistants = assistant_rows(rows)
    evidence = selected_evidence(evidence_dir)
    manifest = write_manifest(rows, selected, assistants, evidence, DEFAULT_OUTPUT_DIR, DEFAULT_ZIP, False, DEFAULT_GENERATED_DATE)
    if "MODEL_PROFILE_386_MIN_LEXICON384_REPAIR" not in manifest:
        raise RuntimeError("manifest does not call out the failed 386-min repair")
    if "MODEL_TOKHEADQ4_PROD_PROBE" not in manifest:
        raise RuntimeError("manifest missing low-memory q4 release model")
    if "ASSISTANT_DOSHELP" not in manifest or "ASSISTANT_OFFICE" not in manifest:
        raise RuntimeError("manifest missing assistant pack models")
    if "SHA256SUMS.txt" not in manifest or "v0.1.0-preview" not in manifest:
        raise RuntimeError("manifest missing release checksum or version fields")
    if "OS/2/Warp package" not in manifest or "DOS demo" not in manifest:
        raise RuntimeError("manifest missing DOS-only preview scope")
    if "qemu/evidence/run_main_486.log" not in manifest:
        raise RuntimeError("manifest missing DOS demo run log")
    home_fragment = "/" + "Users/"
    if home_fragment in sanitize_public_text(str(ROOT)):
        raise RuntimeError("release text sanitizer did not replace local root")
    if sanitize_public_text(f"model_dir: {ROOT}/assets/gpt2_basic/MODEL") != "model_dir: <repo>/assets/gpt2_basic/MODEL":
        raise RuntimeError("release text sanitizer changed paths unexpectedly")
    require_release_sources_tracked(release_source_files(selected, evidence, pack_root))
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = Path(tmp) / "preview"
        package_dir.mkdir()
        (package_dir / "marker.txt").write_text("x\n", encoding="ascii")
        existing_manifest = write_manifest(rows, selected, assistants, evidence, package_dir, DEFAULT_ZIP, True, DEFAULT_GENERATED_DATE)
        if "Package status: `1 files" not in existing_manifest:
            raise RuntimeError("manifest-only mode does not preserve existing package-tree status")
    print(f"PROBE_OK preview_release_models={len(selected)}")
    print(f"PROBE_OK preview_release_assistant_models={len(assistants)}")
    print(f"PROBE_OK preview_release_evidence_files={len(evidence)}")
    print("PROBE_OK preview_release_manifest=1")
    print("PROBE_OK preview_release_checksums=1")
    print("PROBE_OK preview_release_tracked_inputs=1")
    print("PROBE_OK artifact_preview_release_manifest_md=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--zip-sha256", type=Path, default=DEFAULT_ZIP_SHA256)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--generated-date", default=DEFAULT_GENERATED_DATE)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        date.fromisoformat(args.generated_date)
    except ValueError as exc:
        raise SystemExit(f"PREVIEW_RELEASE_FAILED generated_date={args.generated_date}") from exc

    rows = audit_models(args.models_root, args.pack_root, args.evidence_dir, refresh_model_reports=False)
    selected = release_rows(rows)
    assistants = assistant_rows(rows)
    evidence = selected_evidence(args.evidence_dir)

    if args.self_test:
        self_test(rows, args.evidence_dir, args.pack_root)
        return

    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    if not args.manifest_only:
        pending_manifest = write_manifest(rows, selected, assistants, evidence, args.output_dir, args.zip_path, False, args.generated_date)
        build_package(args.output_dir, pending_manifest, selected, evidence, args.pack_root, args.force)
        write_converged_package_manifest(
            rows,
            selected,
            assistants,
            evidence,
            args.output_dir,
            args.zip_path,
            args.manifest,
            args.generated_date,
        )
        if not args.no_zip:
            create_zip(args.output_dir, args.zip_path, args.force)
            write_zip_checksum(args.zip_path, args.zip_sha256)
    else:
        manifest_text = write_manifest(
            rows,
            selected,
            assistants,
            evidence,
            args.output_dir,
            args.zip_path,
            args.output_dir.exists(),
            args.generated_date,
        )
        args.manifest.write_text(manifest_text, encoding="ascii")

    print(f"PREVIEW_RELEASE_MANIFEST|path={args.manifest}")
    if not args.manifest_only:
        print(f"PREVIEW_RELEASE_TREE|path={args.output_dir}")
        if not args.no_zip:
            print(f"PREVIEW_RELEASE_ZIP|path={args.zip_path}")
            print(f"PREVIEW_RELEASE_ZIP_SHA256|path={args.zip_sha256}")


if __name__ == "__main__":
    main()

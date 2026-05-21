#!/usr/bin/env python3
"""Verify and stage physical GPT2-BASIC hardware capture logs."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
from pathlib import Path

try:
    from scripts import verify_hardware_capture
except ImportError:  # pragma: no cover - used when run as scripts/foo.py
    import verify_hardware_capture  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = ROOT / "qemu" / "evidence"
MACHINE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STAGED_FILES = (
    ("capture", "HWVALID.LOG", "capture.log"),
    ("quality", "QUAL.LOG", "quality.log"),
    ("perf", "PERF.LOG", "perf.log"),
    ("assistant", "ASSIST.LOG", "assistant.log"),
    ("assistant_stress", "ASTRESS.LOG", "assistant_stress.log"),
    ("assistant_recall", "ARECALL.LOG", "assistant_recall.log"),
    ("assistant_compile", "ASSISTC.LOG", "assistant_compile.log"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"HARDWARE_STAGE_FAILED {message}")


def validate_machine_key(machine_key: str) -> None:
    require(
        MACHINE_KEY_RE.fullmatch(machine_key) is not None,
        f"invalid_machine_key={machine_key}",
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="ascii", errors="ignore")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def staged_log_name(machine_key: str, suffix: str) -> str:
    return f"hardware_{machine_key}_{suffix}"


def note_fields(notes_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in notes_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            fields[key] = value.strip()
    return fields


def verify_notes_machine_key(notes_src: Path, machine_key: str) -> None:
    fields = note_fields(read_text(notes_src))
    notes_key = fields.get("Machine key", "")
    if notes_key:
        require(notes_key == machine_key, f"machine_key_mismatch=notes:{notes_key},arg:{machine_key}")


def perf_summary(perf_text: str) -> str:
    for line in perf_text.splitlines():
        if line.startswith("PERF_SUMMARY|"):
            return line
    return ""


def markdown_cell(value: str) -> str:
    return value.replace("|", r"\|")


def write_notes_markdown(src: Path, dst: Path, machine_key: str) -> None:
    notes = read_text(src).strip()
    fields = note_fields(notes)
    lines = [
        f"# GPT2-BASIC Hardware Notes: {machine_key}",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for field in (
        "Machine key",
        "CPU",
        "Clock",
        "RAM",
        "DOS version",
        "FreeBASIC version",
        "Storage",
        "Cache/turbo state",
        "Video",
        "Model directory",
        "Pack directory",
    ):
        lines.append(f"| {field} | {markdown_cell(fields.get(field, ''))} |")
    lines.extend(["", "## Raw Notes", "", "```text", notes, "```", ""])
    dst.write_text("\n".join(lines), encoding="ascii")


def write_manifest(
    manifest_path: Path,
    capture_dir: Path,
    evidence_dir: Path,
    machine_key: str,
    staged_paths: list[Path],
) -> None:
    perf_path = evidence_dir / staged_log_name(machine_key, "perf.log")
    summary = perf_summary(read_text(perf_path)) if perf_path.exists() else ""
    lines = [
        f"# GPT2-BASIC Hardware Capture: {machine_key}",
        "",
        f"Capture directory: `{capture_dir}`",
        "",
        "Verification: PASS",
        "",
        "## Staged Files",
        "",
    ]
    for path in staged_paths:
        lines.append(f"- `{path.relative_to(evidence_dir).as_posix()}`")
    lines.extend(["", "## File Checksums", "", "| SHA256 | Bytes | Path |", "|---|---:|---|"])
    for path in staged_paths:
        rel_path = path.relative_to(evidence_dir).as_posix()
        lines.append(f"| `{sha256(path)}` | {path.stat().st_size} | `{rel_path}` |")
    if summary:
        lines.extend(["", "## Performance Summary", "", f"`{summary}`"])
    lines.append("")
    manifest_path.write_text("\n".join(lines), encoding="ascii")


def parse_manifest_checksum_rows(manifest_path: Path) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for raw_line in read_text(manifest_path).splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        digest = cells[0].strip("`")
        size_text = cells[1]
        rel_path = cells[2].strip("`")
        require(SHA256_RE.fullmatch(digest) is not None, f"manifest_bad_sha256={manifest_path}")
        require(size_text.isdecimal(), f"manifest_bad_size={manifest_path}")
        rows.append((digest, int(size_text), rel_path))
    return rows


def verify_staged_manifest(evidence_dir: Path, machine_key: str) -> None:
    validate_machine_key(machine_key)
    evidence_dir = evidence_dir.resolve()
    manifest = evidence_dir / staged_log_name(machine_key, "manifest.md")
    require(manifest.is_file(), f"manifest_missing={manifest}")
    rows = parse_manifest_checksum_rows(manifest)
    require(rows, f"manifest_checksums_missing={manifest}")
    seen: set[str] = set()
    for expected_sha, expected_size, rel_path in rows:
        path = Path(rel_path)
        require(not path.is_absolute() and ".." not in path.parts, f"manifest_bad_path={rel_path}")
        require(rel_path not in seen, f"manifest_duplicate_path={rel_path}")
        seen.add(rel_path)
        staged = evidence_dir / path
        require(staged.is_file(), f"manifest_file_missing={rel_path}")
        require(staged.stat().st_size == expected_size, f"manifest_size_mismatch={rel_path}")
        require(sha256(staged) == expected_sha, f"manifest_sha256_mismatch={rel_path}")


def output_paths(evidence_dir: Path, machine_key: str, include_notes: bool = True) -> list[Path]:
    paths = [
        evidence_dir / staged_log_name(machine_key, suffix)
        for _, _, suffix in STAGED_FILES
    ]
    if include_notes:
        paths.append(evidence_dir / staged_log_name(machine_key, "notes.md"))
    paths.append(evidence_dir / staged_log_name(machine_key, "manifest.md"))
    return paths


def stage_capture(
    capture_dir: Path,
    evidence_dir: Path,
    machine_key: str,
    require_assistant: bool,
    require_notes: bool,
    require_filled_notes: bool,
    force: bool,
) -> list[Path]:
    validate_machine_key(machine_key)
    capture_dir = capture_dir.resolve()
    evidence_dir = evidence_dir.resolve()

    verify_hardware_capture.verify_capture(
        capture_dir,
        require_assistant=require_assistant,
        require_notes=require_notes,
        require_filled_notes=require_filled_notes,
    )

    notes_src = capture_dir / verify_hardware_capture.DEFAULT_FILES["notes"]
    include_notes = notes_src.exists()
    if include_notes:
        verify_notes_machine_key(notes_src, machine_key)
    staged_paths = output_paths(evidence_dir, machine_key, include_notes=include_notes)
    existing = [path for path in staged_paths if path.exists()]
    require(force or not existing, "staged_exists=" + ",".join(path.name for path in existing[:5]))

    evidence_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for _, raw_name, suffix in STAGED_FILES:
        src = capture_dir / raw_name
        if not src.exists():
            continue
        dst = evidence_dir / staged_log_name(machine_key, suffix)
        shutil.copy2(src, dst)
        written.append(dst)

    if include_notes:
        notes_dst = evidence_dir / staged_log_name(machine_key, "notes.md")
        write_notes_markdown(notes_src, notes_dst, machine_key)
        written.append(notes_dst)

    manifest = evidence_dir / staged_log_name(machine_key, "manifest.md")
    write_manifest(manifest, capture_dir, evidence_dir, machine_key, written)
    written.append(manifest)

    for path in written:
        print(f"HARDWARE_STAGE_FILE|path={path}")
    print(f"PROBE_OK hardware_capture_staged={machine_key}")
    return written


def write_sample_capture(root: Path) -> None:
    (root / "HWVALID.LOG").write_text(
        "HW_CAPTURE_BEGIN\nHW_STEP|quality_all\nHW_CAPTURE_END\n",
        encoding="ascii",
    )
    quality_rows = []
    for i in range(10):
        quality_rows.append(f"QUALITY_PROMPT_BEGIN|p{i}|Prompt")
        quality_rows.append(f"QUALITY_PROMPT_END|p{i}")
    (root / "QUAL.LOG").write_text(
        "QUALITY_SUITE_BEGIN|all\n" + "\n".join(quality_rows) + "\nQUALITY_SUITE_END|all\n",
        encoding="ascii",
    )
    (root / "PERF.LOG").write_text(
        "PERF_BEGIN|suite=gpt2-basic-hardware|version=2\n"
        "PERF_MODEL|profile=test\n"
        "PERF_RUN|name=a|tokens_per_sec=1\n"
        "PERF_RUN|name=b|tokens_per_sec=1\n"
        "PERF_RUN|name=c|tokens_per_sec=1\n"
        "PERF_SUMMARY|runs=3|tokens=3|seconds=3|tokens_per_sec=1.0\n"
        "PERF_END\n",
        encoding="ascii",
    )
    (root / "ASSIST.LOG").write_text(
        "ASSIST_BEGIN|suite=pack-shell|version=1\n"
        + "".join(
            f"ASSIST_PACK|id={pack_id}\nASSIST_MODEL|pack={pack_id}\nASSIST_REPLY|pack={pack_id}\n"
            for pack_id in verify_hardware_capture.EXPECTED_ASSISTANT_PACKS
        )
        + f"ASSIST_END|packs={verify_hardware_capture.EXPECTED_ASSISTANT_PACK_COUNT}\n",
        encoding="ascii",
    )
    stress_lines = [
        "ASSIST_BEGIN|suite=stress-probe|version=1",
        *[f"ASSIST_PACK|id={pack_id}" for pack_id in verify_hardware_capture.EXPECTED_ASSISTANT_PACKS],
    ]
    for case in verify_hardware_capture.stress_assistant_behavior.EXPECTED_CASES:
        stress_lines.append(
            f"ASSIST_REPLY|pack={case.pack}|intent=general_chat|ui=text|query={case.query}|"
            f"source=retrieval|recall=kb2_term|answer={case.terms[0]} check passed."
        )
    stress_lines.append(
        f"ASSIST_END|suite=stress-probe|packs={verify_hardware_capture.EXPECTED_ASSISTANT_PACK_COUNT}"
    )
    (root / "ASTRESS.LOG").write_text(
        "\n".join(stress_lines) + "\n",
        encoding="ascii",
    )
    recall_lines = [
        "ASSIST_BEGIN|suite=recall-probe|version=1",
        *[f"ASSIST_PACK|id={pack_id}" for pack_id in verify_hardware_capture.EXPECTED_ASSISTANT_PACKS],
    ]
    for case in verify_hardware_capture.benchmark_assistant_recall.CASES:
        recall_lines.append(
            "ASSIST_RECALL|pack={pack}|query={query}|recall=kb2_term|"
            "recall_score=99|t_retrieve_ms=3|answer={answer}".format(
                pack=case.pack,
                query=case.query,
                answer=" ".join(case.terms) + ".",
            )
        )
    recall_lines.append(
        f"ASSIST_END|suite=recall-probe|packs={verify_hardware_capture.EXPECTED_ASSISTANT_PACK_COUNT}"
    )
    (root / "ARECALL.LOG").write_text(
        "\n".join(recall_lines) + "\n",
        encoding="ascii",
    )
    (root / "ASSISTC.LOG").write_text("ASSIST_COMPILE_OK\n", encoding="ascii")
    (root / "HWNOTES.TXT").write_text(
        "Machine key: 486dx2_66_dos622\n"
        "CPU: 486DX2\n"
        "Clock: 66 MHz\n"
        "RAM: 32 MB\n"
        "DOS version: MS-DOS 6.22\n"
        "FreeBASIC version: 1.10\n"
        "Storage: IDE CF\n"
        "Cache/turbo state: cache on, turbo on\n"
        "Video: VGA\n"
        "Model directory: C:\\GPT2\\MODEL\n"
        "Pack directory: C:\\GPT2\\PACKS\n"
        "Notes: self test\n",
        encoding="ascii",
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        capture = root / "capture"
        evidence = root / "evidence"
        capture.mkdir()
        write_sample_capture(capture)
        written = stage_capture(
            capture,
            evidence,
            "486dx2_66_dos622",
            require_assistant=True,
            require_notes=True,
            require_filled_notes=True,
            force=False,
        )
        require(
            (evidence / "hardware_486dx2_66_dos622_quality.log").exists(),
            "self_test_missing_quality",
        )
        require(
            (evidence / "hardware_486dx2_66_dos622_notes.md").exists(),
            "self_test_missing_notes",
        )
        require(
            (evidence / "hardware_486dx2_66_dos622_manifest.md").exists(),
            "self_test_missing_manifest",
        )
        require(len(written) == 9, "self_test_staged_count")
    print("PROBE_OK hardware_stage_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=Path("."))
    parser.add_argument("--machine-key")
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--allow-missing-assistant", action="store_true")
    parser.add_argument("--allow-missing-notes", action="store_true")
    parser.add_argument("--allow-template-notes", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-staged", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    require(args.machine_key is not None, "missing_machine_key")
    if args.verify_staged:
        verify_staged_manifest(args.evidence_dir, args.machine_key)
        print(f"PROBE_OK hardware_staged_manifest={args.machine_key}")
        return

    stage_capture(
        args.capture_dir,
        args.evidence_dir,
        args.machine_key,
        require_assistant=not args.allow_missing_assistant,
        require_notes=not args.allow_missing_notes,
        require_filled_notes=not args.allow_template_notes,
        force=args.force,
    )


if __name__ == "__main__":
    main()

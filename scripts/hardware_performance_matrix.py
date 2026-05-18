#!/usr/bin/env python3
"""Build the physical-hardware GPT2-BASIC performance matrix from staged logs."""

from __future__ import annotations

import argparse
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts import stage_hardware_capture_evidence
except ImportError:  # pragma: no cover - used when run as scripts/foo.py
    import stage_hardware_capture_evidence  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = ROOT / "qemu" / "evidence"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE_DIR / "hardware_performance_matrix.md"
PERF_LOG_RE = re.compile(r"^hardware_(?P<machine>[a-z0-9][a-z0-9_-]{1,63})_perf\.log$")


@dataclass(frozen=True)
class PerfCapture:
    machine: str
    cpu: str
    clock: str
    ram: str
    cache_state: str
    storage: str
    dos_version: str
    freebasic_version: str
    profile: str
    runtime_bytes: int | None
    runs: int
    tokens: int
    seconds: float
    tokens_per_sec: float
    perf_log: Path
    notes: Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"HARDWARE_MATRIX_FAILED {message}")


def parse_perf_record(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split("|")
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return parts[0], values


def parse_notes_markdown(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip().replace(r"\|", "|") for cell in line.strip("|").split("|")]
        if len(cells) != 2 or cells[0] == "Field":
            continue
        fields[cells[0]] = cells[1]
    return fields


def parse_perf_log(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    model: dict[str, str] = {}
    summary: dict[str, str] = {}
    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("PERF_"):
            continue
        kind, values = parse_perf_record(line)
        if kind == "PERF_MODEL":
            model = values
        elif kind == "PERF_SUMMARY":
            summary = values

    require(model, f"perf_model_missing={path}")
    require(summary, f"perf_summary_missing={path}")
    return model, summary


def capture_from_perf_log(perf_log: Path, require_notes: bool) -> PerfCapture | None:
    match = PERF_LOG_RE.fullmatch(perf_log.name)
    if not match:
        return None

    machine = match.group("machine")
    notes = perf_log.with_name(f"hardware_{machine}_notes.md")
    if not notes.exists():
        require(not require_notes, f"notes_missing={notes}")
        note_fields: dict[str, str] = {}
    else:
        note_fields = parse_notes_markdown(notes)
        notes_machine = note_fields.get("Machine key", "")
        if notes_machine:
            require(notes_machine == machine, f"machine_key_mismatch=notes:{notes_machine},file:{machine}")

    model, summary = parse_perf_log(perf_log)
    return PerfCapture(
        machine=machine,
        cpu=note_fields.get("CPU", ""),
        clock=note_fields.get("Clock", ""),
        ram=note_fields.get("RAM", ""),
        cache_state=note_fields.get("Cache/turbo state", ""),
        storage=note_fields.get("Storage", ""),
        dos_version=note_fields.get("DOS version", ""),
        freebasic_version=note_fields.get("FreeBASIC version", ""),
        profile=model.get("profile", ""),
        runtime_bytes=int(model["runtime_bytes"]) if model.get("runtime_bytes", "").isdigit() else None,
        runs=int(summary.get("runs", "0")),
        tokens=int(summary.get("tokens", "0")),
        seconds=float(summary.get("seconds", "0")),
        tokens_per_sec=float(summary.get("tokens_per_sec", "0")),
        perf_log=perf_log,
        notes=notes,
    )


def find_captures(evidence_dir: Path, require_notes: bool = True) -> list[PerfCapture]:
    captures: list[PerfCapture] = []
    if not evidence_dir.exists():
        return captures
    for perf_log in sorted(evidence_dir.glob("hardware_*_perf.log")):
        capture = capture_from_perf_log(perf_log, require_notes=require_notes)
        if capture is not None:
            captures.append(capture)
    return captures


def fmt_float(value: float) -> str:
    return f"{value:.2f}"


def fmt_int(value: int | None) -> str:
    return "" if value is None else str(value)


def markdown_cell(value: str) -> str:
    return value.replace("|", r"\|")


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_report(captures: list[PerfCapture], output: Path, evidence_dir: Path) -> None:
    lines = [
        "# GPT2-BASIC Physical Hardware Performance Matrix",
        "",
        "This report is generated only from staged physical capture logs named",
        "`hardware_<machine>_perf.log`. It deliberately does not read QEMU",
        "`perf_486_*` logs, host-speed measurements, or planning estimates.",
        "",
    ]
    if not captures:
        lines.extend(
            [
                "No staged physical hardware performance logs were found yet.",
                "",
                "Run the DOS hardware capture on a real machine, copy the logs back,",
                "then stage them with:",
                "",
                "```sh",
                "python3 scripts/stage_hardware_capture_evidence.py \\",
                "  --capture-dir /path/to/capture \\",
                "  --machine-key 486dx2_66_dos622",
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Machine | CPU | Clock | RAM | Cache/Turbo | Storage | DOS | FreeBASIC | Model profile | Runtime bytes | Runs | Tokens | Seconds | Tok/s | Evidence |",
                "|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for capture in captures:
            evidence = rel(capture.perf_log, evidence_dir.parent.parent)
            lines.append(
                "| "
                + " | ".join(
                    [
                        capture.machine,
                        markdown_cell(capture.cpu),
                        markdown_cell(capture.clock),
                        markdown_cell(capture.ram),
                        markdown_cell(capture.cache_state),
                        markdown_cell(capture.storage),
                        markdown_cell(capture.dos_version),
                        markdown_cell(capture.freebasic_version),
                        markdown_cell(capture.profile),
                        fmt_int(capture.runtime_bytes),
                        str(capture.runs),
                        str(capture.tokens),
                        fmt_float(capture.seconds),
                        fmt_float(capture.tokens_per_sec),
                        f"`{evidence}`",
                    ]
                )
                + " |"
            )
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="ascii")


def build_matrix(evidence_dir: Path, output: Path, require_notes: bool) -> list[PerfCapture]:
    captures = find_captures(evidence_dir, require_notes=require_notes)
    write_report(captures, output, evidence_dir)
    print(f"HARDWARE_PERFORMANCE_MATRIX|path={output}")
    print(f"PROBE_OK hardware_performance_rows={len(captures)}")
    return captures


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        capture_dir = root / "capture"
        evidence_dir = root / "evidence"
        output = root / "matrix.md"
        capture_dir.mkdir()
        stage_hardware_capture_evidence.write_sample_capture(capture_dir)
        stage_hardware_capture_evidence.stage_capture(
            capture_dir,
            evidence_dir,
            "486dx2_66_dos622",
            require_assistant=True,
            require_notes=True,
            require_filled_notes=True,
            force=False,
        )
        captures = build_matrix(evidence_dir, output, require_notes=True)
        require(len(captures) == 1, "self_test_row_count")
        text = output.read_text(encoding="ascii")
        require("486dx2_66_dos622" in text, "self_test_missing_machine")
        require("1.00" in text, "self_test_missing_tokens_per_sec")
    print("PROBE_OK hardware_performance_matrix_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-missing-notes", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    build_matrix(
        args.evidence_dir,
        args.output,
        require_notes=not args.allow_missing_notes,
    )


if __name__ == "__main__":
    main()

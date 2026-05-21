#!/usr/bin/env python3
"""Verify logs captured from a physical GPT2-BASIC DOS run."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

try:
    from scripts import stress_assistant_behavior
except ImportError:  # pragma: no cover - used when run as scripts/foo.py
    import stress_assistant_behavior  # type: ignore


DEFAULT_FILES = {
    "capture": "HWVALID.LOG",
    "quality": "QUAL.LOG",
    "perf": "PERF.LOG",
    "assistant": "ASSIST.LOG",
    "assistant_stress": "ASTRESS.LOG",
    "assistant_compile": "ASSISTC.LOG",
    "notes": "HWNOTES.TXT",
}
EXPECTED_ASSISTANT_PACKS = ("CHAT", "DOSHELP", "OFFICE", "DEV", "PORTABLE")
EXPECTED_ASSISTANT_PACK_COUNT = len(EXPECTED_ASSISTANT_PACKS)
EXPECTED_STRESS_REPLIES = len(stress_assistant_behavior.EXPECTED_CASES)
NOTE_FIELDS = (
    "Machine key:",
    "CPU:",
    "Clock:",
    "RAM:",
    "DOS version:",
    "FreeBASIC version:",
    "Storage:",
    "Cache/turbo state:",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"HARDWARE_CAPTURE_FAILED {message}")


def read(path: Path, required: bool = True) -> str:
    if not path.exists():
        require(not required, f"missing={path}")
        return ""
    return path.read_text(encoding="ascii", errors="ignore")


def count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def verify_capture_log(path: Path) -> None:
    text = read(path)
    require("HW_CAPTURE_BEGIN" in text, "capture_begin_missing")
    require("HW_CAPTURE_END" in text, "capture_end_missing")
    require("HW_CAPTURE_FAILED" not in text, "capture_failed_marker_present")


def verify_quality_log(path: Path) -> int:
    text = read(path)
    require("QUALITY_SUITE_BEGIN|all" in text, "quality_suite_missing")
    require("QUALITY_SUITE_END|all" in text, "quality_suite_end_missing")
    begin_count = count_matches(r"^QUALITY_PROMPT_BEGIN\|", text)
    end_count = count_matches(r"^QUALITY_PROMPT_END\|", text)
    require(begin_count >= 10, f"quality_prompt_count={begin_count}")
    require(begin_count == end_count, f"quality_prompt_mismatch={begin_count}/{end_count}")
    require("MODEL_FAILED" not in text and "QUALITY_FAILED" not in text, "quality_failure_marker_present")
    return begin_count


def verify_perf_log(path: Path) -> int:
    text = read(path)
    require("PERF_BEGIN|suite=gpt2-basic-hardware|version=2" in text, "perf_begin_missing")
    require("PERF_MODEL|" in text, "perf_model_missing")
    require("PERF_SUMMARY|" in text, "perf_summary_missing")
    require("PERF_END" in text, "perf_end_missing")
    run_count = count_matches(r"^PERF_RUN\|", text)
    require(run_count >= 3, f"perf_run_count={run_count}")
    summary = re.search(r"PERF_SUMMARY\|.*tokens_per_sec=([0-9.]+)", text)
    require(summary is not None, "perf_tokens_per_sec_missing")
    require(float(summary.group(1)) > 0.0, "perf_tokens_per_sec_nonpositive")
    return run_count


def verify_assistant_log(path: Path, required: bool) -> int:
    text = read(path, required=required)
    if not text:
        return 0
    require("ASSIST_BEGIN|suite=pack-shell|version=1" in text, "assistant_begin_missing")
    require(f"ASSIST_END|packs={EXPECTED_ASSISTANT_PACK_COUNT}" in text, "assistant_end_missing")
    for pack_id in EXPECTED_ASSISTANT_PACKS:
        require(f"ASSIST_PACK|id={pack_id}" in text, f"assistant_pack_missing={pack_id}")
        require(f"ASSIST_MODEL|pack={pack_id}" in text, f"assistant_model_missing={pack_id}")
        require(f"ASSIST_REPLY|pack={pack_id}" in text, f"assistant_reply_missing={pack_id}")
    return count_matches(r"^ASSIST_REPLY\|", text)


def verify_assistant_stress_log(path: Path, required: bool) -> int:
    text = read(path, required=required)
    if not text:
        return 0
    require("ASSIST_BEGIN|suite=stress-probe|version=1" in text, "assistant_stress_begin_missing")
    require(
        f"ASSIST_END|suite=stress-probe|packs={EXPECTED_ASSISTANT_PACK_COUNT}" in text,
        "assistant_stress_end_missing",
    )
    require("status=model_unavailable" not in text, "assistant_stress_model_unavailable")
    require("|query=" in text and "|answer=" in text, "assistant_stress_structured_answer_missing")
    for pack_id in EXPECTED_ASSISTANT_PACKS:
        require(f"ASSIST_PACK|id={pack_id}" in text, f"assistant_stress_pack_missing={pack_id}")
        require(f"ASSIST_REPLY|pack={pack_id}" in text, f"assistant_stress_reply_missing={pack_id}")
    reply_count = count_matches(r"^ASSIST_REPLY\|", text)
    require(reply_count == EXPECTED_STRESS_REPLIES, f"assistant_stress_reply_count={reply_count}")
    records = stress_assistant_behavior.parse_records(text)
    stress_assistant_behavior.validate_records(records)
    return reply_count


def verify_assistant_compile_log(path: Path, required: bool) -> bool:
    text = read(path, required=required)
    if not text:
        return False
    require("ASSIST_COMPILE_FAILED" not in text, "assistant_compile_failed")
    require("ASSIST_COMPILE_OK" in text, "assistant_compile_ok_missing")
    return True


def note_field_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        values[f"{key.strip()}:"] = value.strip()
    return values


def verify_notes(path: Path, required: bool, require_values: bool) -> bool:
    text = read(path, required=required)
    if not text:
        return False
    values = note_field_values(text)
    for field in NOTE_FIELDS:
        require(field in values, f"notes_field_missing={field.rstrip(':')}")
        if require_values:
            require(values[field] != "", f"notes_field_empty={field.rstrip(':')}")
    return True


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "HWVALID.LOG").write_text("HW_CAPTURE_BEGIN\nHW_STEP|quality_all\nHW_CAPTURE_END\n", encoding="ascii")
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
                for pack_id in EXPECTED_ASSISTANT_PACKS
            )
            + f"ASSIST_END|packs={EXPECTED_ASSISTANT_PACK_COUNT}\n",
            encoding="ascii",
        )
        stress_lines = [
            "ASSIST_BEGIN|suite=stress-probe|version=1",
            *[f"ASSIST_PACK|id={pack_id}" for pack_id in EXPECTED_ASSISTANT_PACKS],
        ]
        for case in stress_assistant_behavior.EXPECTED_CASES:
            stress_lines.append(
                f"ASSIST_REPLY|pack={case.pack}|intent=general_chat|ui=text|query={case.query}|"
                f"source=retrieval|recall=kb2_term|answer={case.terms[0]} check passed."
            )
        stress_lines.append(f"ASSIST_END|suite=stress-probe|packs={EXPECTED_ASSISTANT_PACK_COUNT}")
        (root / "ASTRESS.LOG").write_text(
            "\n".join(stress_lines) + "\n",
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
            "Cache/turbo state: cache on, turbo on\n",
            encoding="ascii",
        )
        verify_capture(root, require_assistant=True, require_notes=True, require_filled_notes=True)
    print("PROBE_OK hardware_capture_self_test=1")
    print("PROBE_OK hardware_capture_quality_gate=1")
    print("PROBE_OK hardware_capture_perf_gate=1")
    print("PROBE_OK hardware_capture_assistant_gate=1")


def verify_capture(
    capture_dir: Path,
    require_assistant: bool,
    require_notes: bool,
    require_filled_notes: bool = False,
) -> None:
    verify_capture_log(capture_dir / DEFAULT_FILES["capture"])
    quality_count = verify_quality_log(capture_dir / DEFAULT_FILES["quality"])
    perf_count = verify_perf_log(capture_dir / DEFAULT_FILES["perf"])
    assistant_count = verify_assistant_log(capture_dir / DEFAULT_FILES["assistant"], require_assistant)
    assistant_stress_count = verify_assistant_stress_log(
        capture_dir / DEFAULT_FILES["assistant_stress"],
        require_assistant,
    )
    assistant_compiled = verify_assistant_compile_log(capture_dir / DEFAULT_FILES["assistant_compile"], require_assistant)
    notes_present = verify_notes(
        capture_dir / DEFAULT_FILES["notes"],
        require_notes,
        require_values=require_filled_notes,
    )

    print(f"PROBE_OK hardware_capture_log={DEFAULT_FILES['capture']}")
    print(f"PROBE_OK hardware_quality_log={DEFAULT_FILES['quality']}")
    print(f"PROBE_OK hardware_perf_log={DEFAULT_FILES['perf']}")
    if require_assistant or assistant_count:
        print(f"PROBE_OK hardware_assistant_log={DEFAULT_FILES['assistant']}")
    if require_assistant or assistant_stress_count:
        print(f"PROBE_OK hardware_assistant_stress_log={DEFAULT_FILES['assistant_stress']}")
    if require_assistant or assistant_compiled:
        print(f"PROBE_OK hardware_assistant_compile_log={DEFAULT_FILES['assistant_compile']}")
    if require_notes or notes_present:
        print(f"PROBE_OK hardware_notes_template={DEFAULT_FILES['notes']}")
    print(f"PROBE_OK hardware_quality_prompts={quality_count}")
    print(f"PROBE_OK hardware_perf_runs={perf_count}")
    if require_assistant or assistant_count:
        print(f"PROBE_OK hardware_assistant_replies={assistant_count}")
    if require_assistant or assistant_stress_count:
        print(f"PROBE_OK hardware_assistant_stress_replies={assistant_stress_count}")
    if require_assistant or assistant_compiled:
        print("PROBE_OK hardware_assistant_compile=1")
    if require_notes or notes_present:
        print("PROBE_OK hardware_notes=1")
    print("PROBE_OK hardware_capture=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=Path("."))
    parser.add_argument("--allow-missing-assistant", action="store_true")
    parser.add_argument("--allow-missing-notes", action="store_true")
    parser.add_argument("--require-filled-notes", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    verify_capture(
        args.capture_dir,
        require_assistant=not args.allow_missing_assistant,
        require_notes=not args.allow_missing_notes,
        require_filled_notes=args.require_filled_notes,
    )


if __name__ == "__main__":
    main()

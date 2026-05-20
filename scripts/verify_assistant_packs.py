#!/usr/bin/env python3
"""Verify the GPT2-BASIC assistant-pack surface and QEMU evidence."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK_ROOT = ROOT / "assets" / "gpt2_basic" / "PACKS"
DEFAULT_ASSISTANT_LOG = ROOT / "qemu" / "evidence" / "assistant_486.log"
DEFAULT_COMPILE_LOG = ROOT / "qemu" / "evidence" / "assistant_compile_486.log"
DEFAULT_EVIDENCE_DIR = ROOT / "qemu" / "evidence"
DEFAULT_STRESS_LOG = ROOT / "qemu" / "evidence" / "assistant_stress_486.log"
DEFAULT_STRESS_COMPILE_LOG = ROOT / "qemu" / "evidence" / "assistant_stress_compile_486.log"
DEFAULT_STRESS_REPORT = ROOT / "qemu" / "evidence" / "assistant_stress_report.md"
DEFAULT_RAW_PROMPT_REPORT = ROOT / "qemu" / "evidence" / "assistant_raw_prompt_eval.md"


@dataclass(frozen=True)
class PackInfo:
    pack_id: str
    model_path: str
    pack_dir: Path
    usage_path: Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ASSISTANT_PROBE_FAILED {message}")


def read(path: Path) -> str:
    require(path.exists(), f"missing={path}")
    return path.read_text(encoding="ascii", errors="ignore")


def pack_ids(pack_root: Path) -> list[str]:
    list_path = pack_root / "PACKS.TXT"
    text = read(list_path)
    ids: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        ids.append(line.upper())
    return ids


def parse_ini(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip()
    return values


def model_host_path(pack_root: Path, model_value: str) -> Path:
    value = model_value.replace("\\", "/")
    if value.upper().startswith("PACKS/"):
        return pack_root.parent / value
    return ROOT / "assets" / "gpt2_basic" / value


def verify_pack_model(pack_root: Path, pack_id: str, model_value: str) -> None:
    model_dir = model_host_path(pack_root, model_value)
    for name in ("GPT2CFG.TXT", "GPT2WT.BIN", "GPT2FX.BIN", "GPT2EXP.BIN", "PROFILE.TXT"):
        require((model_dir / name).exists(), f"missing_pack_model_{name}={pack_id}")


def verify_pack_files(pack_root: Path) -> list[PackInfo]:
    ids = pack_ids(pack_root)
    require(len(ids) >= 3, "pack_count_lt_3")
    require(ids[0] == "CHAT", "chat_pack_not_default")
    for expected in ("CHAT", "DOSHELP", "OFFICE"):
        require(expected in ids, f"missing_pack={expected}")
    packs: list[PackInfo] = []
    for pack_id in ids:
        pack_dir = pack_root / pack_id
        ini = read(pack_dir / "PACK.INI")
        values = parse_ini(ini)
        help_text = read(pack_dir / "HELP.TXT")
        require(f"ID={pack_id}" in ini.upper(), f"pack_id_mismatch={pack_id}")
        for key in ("TITLE=", "MODEL=", "PERSONA=", "HELP=", "USAGE=", "SPRITE=", "ICONS=", "ACTIONS="):
            require(key in ini.upper(), f"missing_{key.rstrip('=')}={pack_id}")
        require("|" in help_text, f"missing_retrieval_rows={pack_id}")
        usage_text = read(pack_dir / values["USAGE"])
        for marker in ("Purpose:", "How it works:", "How to use it:", "Good prompts:", "Actions:"):
            require(marker in usage_text, f"missing_usage_marker_{marker.rstrip(':').lower().replace(' ', '_')}={pack_id}")
        verify_pack_model(pack_root, pack_id, values["MODEL"])
        packs.append(PackInfo(pack_id, values["MODEL"], pack_dir, pack_dir / values["USAGE"]))
    return packs


def verify_source() -> None:
    source = read(ROOT / "src" / "assistant.bas")
    for needle in (
        "ASSIST_PACK_LIST",
        "AssistLoadPackList",
        "AssistInitializeModel",
        "GPT2BasicLoadModel",
        "ASSIST_PACK",
        "ASSIST_MODEL",
        "ASSIST_REPLY",
        "USAGE",
        "AssistPrintPackUsage",
        "AssistStreamGenerate",
        "AssistCleanGeneratedText",
        "AssistGeneratedLooksBad",
        "AssistFallbackReply",
        "AssistGoldenReply",
        "AssistGuardProbe",
        "AssistStressProbe",
        "AssistPrepareGenerationPrompt",
        "AssistPrefillPrompt",
        "AssistVisibleToken",
        "GPT2BasicPrefillToken",
        "keep_count = context_limit - reserve_count",
        'PRINT "Thinking: ";',
        'PRINT "  ctx"; i + 1; ": ";',
        "PRINT AssistVisibleToken(input_tokens(i));",
        'PRINT "Thinking: sampling output tokens"',
        'progress_text = "<t" + LTRIM$(STR$(i + 1)) + ">"',
        'PRINT "Answer: ";',
        "FOR erase_idx = 1 TO LEN(progress_text)",
        'PRINT CHR$(8); " "; CHR$(8);',
        "Decode(generated_tokens(), generated_count)",
        "CONST ASSIST_MAX_REPLY_TOKENS = 64",
        "CONST ASSIST_SENTENCE_STOP_MIN_TOKENS = 10",
        'INSTR(lower_text, " user:")',
        'INSTR(lower_text, "use two brief sentences")',
        'INSTR(lower_text, "use to brief sentences")',
        'prompt = "User: " + query',
        'command_line = "--stress-probe"',
        '"|query=" + AssistSafeText(query)',
        '"|answer=" + AssistSafeText(bubble)',
        'LCASE$(command_text) = "/about"',
        'LCASE$(command_text) = "/u"',
        'LCASE$(command_text) = "/d"',
        'LCASE$(command_text) = "/h"',
        "SPRITE",
        "ICONS",
        "ASSIST_BEGIN|suite=stress-probe|version=1",
    ):
        require(needle in source, f"source_missing={needle}")


def verify_pack_quality(packs: list[PackInfo], evidence_dir: Path, raw_prompt_report: Path) -> None:
    pack_by_id = {pack.pack_id: pack for pack in packs}
    raw_report = read(raw_prompt_report)
    require("Status: `PASS`" in raw_report, "raw_prompt_eval_not_pass")
    require("Prompt pass rate: `26/26`" in raw_report, "raw_prompt_eval_pass_rate")
    for pack_id in ("CHAT", "DOSHELP", "OFFICE"):
        pack = pack_by_id[pack_id]
        report = read(evidence_dir / f"quality_report_assistant_{pack.pack_id.lower()}.md")
        match = re.search(r"Prompt pass rate:\s+`(\d+)/(\d+)`", report)
        require(match is not None, f"pack_quality_pass_rate_missing={pack.pack_id}")
        passed = int(match.group(1))
        total = int(match.group(2))
        if pack_id == "CHAT":
            require(total > 0 and passed / total >= 0.80, f"pack_quality_pass_rate={pack.pack_id}:{passed}/{total}")
        else:
            require("Quality status: `PASS`" in report, f"pack_quality_not_pass={pack.pack_id}")
            require(total > 0 and passed == total, f"pack_quality_pass_rate={pack.pack_id}:{passed}/{total}")


def verify_qemu_logs(packs: list[PackInfo], assistant_log: Path, compile_log: Path) -> None:
    assist = read(assistant_log)
    compile_text = read(compile_log)
    require("ASSIST_COMPILE_OK" in compile_text, "compile_marker_missing")
    require("ASSIST_BEGIN|suite=pack-shell|version=1" in assist, "begin_marker_missing")
    require("ASSIST_END|packs=" in assist, "end_marker_missing")
    for pack in packs:
        require(f"ASSIST_PACK|id={pack.pack_id}" in assist, f"pack_log_missing={pack.pack_id}")
        require(f"ASSIST_MODEL|pack={pack.pack_id}" in assist, f"model_log_missing={pack.pack_id}")
        require(
            f"path={pack.model_path}" in assist,
            f"model_path_log_missing={pack.pack_id}",
        )
    require("CHAT pack" in assist, "chat_usage_missing")
    require("DOSHELP pack" in assist, "doshelp_usage_missing")
    require("OFFICE pack" in assist, "office_usage_missing")
    require("You are a small friendly DOS conversation assistant. User:" not in assist, "chat_prompt_echo_in_log")
    require("ASSIST_REPLY|pack=CHAT|intent=general_chat" in assist, "chat_reply_missing")
    require("ASSIST_REPLY|pack=DOSHELP|intent=dos_memory" in assist, "doshelp_reply_missing")
    require("ASSIST_REPLY|pack=OFFICE|intent=office_rewrite" in assist, "office_reply_missing")
    require("status=model_unavailable" not in assist, "model_unavailable_in_assistant_log")


def verify_stress_logs(stress_log: Path, stress_compile_log: Path, stress_report: Path) -> None:
    stress = read(stress_log)
    compile_text = read(stress_compile_log)
    report = read(stress_report)
    require("ASSIST_COMPILE_OK" in compile_text, "stress_compile_marker_missing")
    require("ASSIST_BEGIN|suite=stress-probe|version=1" in stress, "stress_begin_marker_missing")
    require("ASSIST_END|suite=stress-probe|packs=3" in stress, "stress_end_marker_missing")
    require(stress.count("ASSIST_REPLY|") == 18, "stress_reply_count_mismatch")
    require("status=model_unavailable" not in stress, "model_unavailable_in_stress_log")
    require("|query=" in stress and "|answer=" in stress, "stress_structured_answer_missing")
    require("Status: `PASS`" in report, "stress_report_not_pass")
    require("Reply count: `18`" in report, "stress_report_reply_count")
    require("Source counts:" in report, "stress_report_source_counts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--assistant-log", type=Path, default=DEFAULT_ASSISTANT_LOG)
    parser.add_argument("--compile-log", type=Path, default=DEFAULT_COMPILE_LOG)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--stress-log", type=Path, default=DEFAULT_STRESS_LOG)
    parser.add_argument("--stress-compile-log", type=Path, default=DEFAULT_STRESS_COMPILE_LOG)
    parser.add_argument("--stress-report", type=Path, default=DEFAULT_STRESS_REPORT)
    parser.add_argument("--raw-prompt-report", type=Path, default=DEFAULT_RAW_PROMPT_REPORT)
    args = parser.parse_args()

    packs = verify_pack_files(args.pack_root)
    verify_source()
    verify_pack_quality(packs, args.evidence_dir, args.raw_prompt_report)
    verify_qemu_logs(packs, args.assistant_log, args.compile_log)
    verify_stress_logs(args.stress_log, args.stress_compile_log, args.stress_report)
    print(f"PROBE_OK assistant_pack_count={len(packs)}")
    print("PROBE_OK assistant_pack_loader=1")
    print("PROBE_OK assistant_pack_models=1")
    print("PROBE_OK assistant_pack_quality=1")
    print("PROBE_OK assistant_model_switch=1")
    print("PROBE_OK assistant_structured_reply=1")
    print("PROBE_OK assistant_art_slots=1")
    print("PROBE_OK assistant_qemu_evidence=1")
    print("PROBE_OK assistant_stress_evidence=1")


if __name__ == "__main__":
    main()

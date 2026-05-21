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
DEFAULT_GENERALIST_PROMPT_REPORT = ROOT / "qemu" / "evidence" / "assistant_generalist_prompt_eval.md"
DEFAULT_RETRIEVAL_REPORT = ROOT / "qemu" / "evidence" / "assistant_pack_retrieval_eval.md"
DEFAULT_USEFULNESS_REPORT = ROOT / "qemu" / "evidence" / "assistant_usefulness_eval.md"
DEFAULT_KDB_INDEX_REPORT = ROOT / "qemu" / "evidence" / "assistant_kdb_index_eval.md"
DEFAULT_KDB_BINARY_REPORT = ROOT / "qemu" / "evidence" / "assistant_kdb_binary_eval.md"
DEFAULT_KDB_TERM_INDEX_REPORT = ROOT / "qemu" / "evidence" / "assistant_kdb_term_index_eval.md"
RAW_PROMPT_MIN_CASES = 83
GENERALIST_PROMPT_MIN_CASES = 24
RETRIEVAL_MIN_CASES = 42
USEFULNESS_MIN_CASES = 37
USEFULNESS_MIN_WORKFLOWS = 9
KDB_INDEX_MIN_CASES = 42
KDB_INDEX_MAX_SCAN_RATIO = 0.65
KDB_BINARY_MIN_CASES = 42
KDB_BINARY_MAX_SCAN_RATIO = 0.65
KDB_TERM_INDEX_MIN_CASES = 42
KDB_TERM_INDEX_MAX_ROW_RATIO = 0.35
STRESS_REPLY_COUNT = 50


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
    for expected in ("CHAT", "DOSHELP", "OFFICE", "DEV", "PORTABLE"):
        require(expected in ids, f"missing_pack={expected}")
    packs: list[PackInfo] = []
    for pack_id in ids:
        pack_dir = pack_root / pack_id
        ini = read(pack_dir / "PACK.INI")
        values = parse_ini(ini)
        help_text = read(pack_dir / "HELP.TXT")
        knowledge_text = read(pack_dir / "KNOW.TXT")
        kdb_text = read(pack_dir / "KDB.TXT")
        kdb_index_text = read(pack_dir / "KDBIDX.TXT")
        kdb_bin = pack_dir / "KB2ALL.BIN"
        kdb_bin_index_text = read(pack_dir / "KB2IDX.TXT")
        kdb_term_index_files = sorted(pack_dir.glob("KB2TERM.TXT"))
        read(pack_dir / "USER.TXT")
        require(f"ID={pack_id}" in ini.upper(), f"pack_id_mismatch={pack_id}")
        for key in ("TITLE=", "MODEL=", "PERSONA=", "HELP=", "KNOW=", "KDB=", "KDBIDX=", "KDBBIN=", "KDBBIDX=", "USER=", "USAGE=", "SPRITE=", "ICONS=", "ACTIONS="):
            require(key in ini.upper(), f"missing_{key.rstrip('=')}={pack_id}")
        require("|" in help_text, f"missing_retrieval_rows={pack_id}")
        require("|" in knowledge_text, f"missing_knowledge_rows={pack_id}")
        require("|" in kdb_text, f"missing_kdb_rows={pack_id}")
        require("|" in kdb_index_text and "KDB" in kdb_index_text, f"missing_kdb_index_rows={pack_id}")
        require(kdb_bin.exists() and kdb_bin.stat().st_size > 64, f"missing_kdb_binary={pack_id}")
        require(kdb_bin.read_bytes().startswith(b"KDB2|V=1|"), f"bad_kdb_binary_header={pack_id}")
        require("|" in kdb_bin_index_text and "KB2" in kdb_bin_index_text, f"missing_kdb_binary_index_rows={pack_id}")
        require(kdb_term_index_files, f"missing_kdb_term_index={pack_id}")
        require(any("|" in read(path) for path in kdb_term_index_files), f"empty_kdb_term_index={pack_id}")
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
        "AssistPrintCapabilities",
        "AssistPrintLimits",
        "AssistPrintSources",
        "AssistStreamGenerate",
        "AssistCleanGeneratedText",
        "AssistGeneratedLooksBad",
        "AssistFallbackReply",
        "AssistGoldenReply",
        "AssistMemoryReply",
        "AssistMemoryContext",
        "AssistScanRetrievalFile",
        "AssistScanBucketedKdb",
        "AssistKdbBucketPath",
        "AssistScanBucketedKdbV2",
        "AssistScanBinaryKdbTermIndex",
        "AssistScanBinaryKdbFile",
        "AssistKdbV2BucketPath",
        "AssistKdbV2TermPath",
        "AssistRetrievalScore",
        "knowledge_path",
        "kdb_path",
        "kdb_index_path",
        "kdb_bin_path",
        "kdb_bin_index_path",
        "user_path",
        "ASSIST_MEMORY_FILE",
        "AssistLoadMemoryFacts",
        "AssistSaveMemoryFacts",
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
        "AssistCanonicalQuery",
        'prompt = "User: " + canonical_query',
        'command_line = "--stress-probe"',
        '"|query=" + AssistSafeText(query)',
        '"|canonical=" + AssistSafeText(canonical_query)',
        '"|memory=" + AssistSafeText(memory_context)',
        '"|recall=" + AssistSafeText(g_assist_last_recall_mode)',
        '"|t_total_ms="',
        '"|answer=" + AssistSafeText(bubble)',
        'reply_source = "memory"',
        'LCASE$(command_text) = "/memory"',
        'LCASE$(command_text) = "/forget"',
        'LEFT$(LCASE$(command_text), 10) = "/remember "',
        "Memory persists in ",
        "USER.TXT",
        'LCASE$(command_text) = "/about"',
        'LCASE$(command_text) = "/capabilities"',
        'LCASE$(command_text) = "/limits"',
        'LCASE$(command_text) = "/sources"',
        'LCASE$(command_text) = "/u"',
        'LCASE$(command_text) = "/d"',
        'LCASE$(command_text) = "/h"',
        "SPRITE",
        "ICONS",
        "ASSIST_BEGIN|suite=stress-probe|version=1",
    ):
        require(needle in source, f"source_missing={needle}")


def verify_pack_quality(
    packs: list[PackInfo],
    evidence_dir: Path,
    raw_prompt_report: Path,
    generalist_prompt_report: Path,
    retrieval_report: Path,
    usefulness_report: Path,
    kdb_index_report: Path,
    kdb_binary_report: Path,
    kdb_term_index_report: Path,
) -> None:
    pack_by_id = {pack.pack_id: pack for pack in packs}
    raw_report = read(raw_prompt_report)
    require("Status: `PASS`" in raw_report, "raw_prompt_eval_not_pass")
    raw_match = re.search(r"Prompt pass rate:\s+`(\d+)/(\d+)`", raw_report)
    require(raw_match is not None, "raw_prompt_eval_pass_rate_missing")
    raw_passed = int(raw_match.group(1))
    raw_total = int(raw_match.group(2))
    require(raw_total >= RAW_PROMPT_MIN_CASES and raw_passed == raw_total, f"raw_prompt_eval_pass_rate={raw_passed}/{raw_total}")
    generalist_report = read(generalist_prompt_report)
    require("Status: `PASS`" in generalist_report, "generalist_prompt_eval_not_pass")
    generalist_match = re.search(r"Prompt pass rate:\s+`(\d+)/(\d+)`", generalist_report)
    require(generalist_match is not None, "generalist_prompt_eval_pass_rate_missing")
    generalist_passed = int(generalist_match.group(1))
    generalist_total = int(generalist_match.group(2))
    require(
        generalist_total >= GENERALIST_PROMPT_MIN_CASES and generalist_passed == generalist_total,
        f"generalist_prompt_eval_pass_rate={generalist_passed}/{generalist_total}",
    )
    retrieval_text = read(retrieval_report)
    require("Status: `PASS`" in retrieval_text, "pack_retrieval_eval_not_pass")
    retrieval_match = re.search(r"Retrieval pass rate:\s+`(\d+)/(\d+)`", retrieval_text)
    require(retrieval_match is not None, "pack_retrieval_eval_pass_rate_missing")
    retrieval_passed = int(retrieval_match.group(1))
    retrieval_total = int(retrieval_match.group(2))
    require(
        retrieval_total >= RETRIEVAL_MIN_CASES and retrieval_passed == retrieval_total,
        f"pack_retrieval_eval_pass_rate={retrieval_passed}/{retrieval_total}",
    )
    usefulness_text = read(usefulness_report)
    require("Status: `PASS`" in usefulness_text, "assistant_usefulness_eval_not_pass")
    usefulness_match = re.search(r"Task pass rate:\s+`(\d+)/(\d+)`", usefulness_text)
    require(usefulness_match is not None, "assistant_usefulness_eval_pass_rate_missing")
    usefulness_passed = int(usefulness_match.group(1))
    usefulness_total = int(usefulness_match.group(2))
    workflow_match = re.search(r"Workflow coverage:\s+`(\d+)/(\d+)`", usefulness_text)
    require(workflow_match is not None, "assistant_usefulness_eval_workflow_coverage_missing")
    workflows_passed = int(workflow_match.group(1))
    workflows_total = int(workflow_match.group(2))
    require(
        usefulness_total >= USEFULNESS_MIN_CASES and usefulness_passed == usefulness_total,
        f"assistant_usefulness_eval_pass_rate={usefulness_passed}/{usefulness_total}",
    )
    require(
        workflows_total >= USEFULNESS_MIN_WORKFLOWS and workflows_passed == workflows_total,
        f"assistant_usefulness_eval_workflows={workflows_passed}/{workflows_total}",
    )
    kdb_index_text = read(kdb_index_report)
    require("Status: `PASS`" in kdb_index_text, "kdb_index_eval_not_pass")
    kdb_index_match = re.search(r"Indexed recall pass rate:\s+`(\d+)/(\d+)`", kdb_index_text)
    require(kdb_index_match is not None, "kdb_index_eval_pass_rate_missing")
    kdb_index_passed = int(kdb_index_match.group(1))
    kdb_index_total = int(kdb_index_match.group(2))
    scan_ratio_match = re.search(r"Candidate scan ratio:\s+`([0-9.]+)`", kdb_index_text)
    require(scan_ratio_match is not None, "kdb_index_eval_scan_ratio_missing")
    scan_ratio = float(scan_ratio_match.group(1))
    require(
        kdb_index_total >= KDB_INDEX_MIN_CASES and kdb_index_passed == kdb_index_total,
        f"kdb_index_eval_pass_rate={kdb_index_passed}/{kdb_index_total}",
    )
    require(scan_ratio <= KDB_INDEX_MAX_SCAN_RATIO, f"kdb_index_scan_ratio={scan_ratio:.3f}")
    kdb_binary_text = read(kdb_binary_report)
    require("Status: `PASS`" in kdb_binary_text, "kdb_binary_eval_not_pass")
    kdb_binary_match = re.search(r"Binary recall pass rate:\s+`(\d+)/(\d+)`", kdb_binary_text)
    require(kdb_binary_match is not None, "kdb_binary_eval_pass_rate_missing")
    kdb_binary_passed = int(kdb_binary_match.group(1))
    kdb_binary_total = int(kdb_binary_match.group(2))
    binary_scan_ratio_match = re.search(r"Candidate row scan ratio:\s+`([0-9.]+)`", kdb_binary_text)
    require(binary_scan_ratio_match is not None, "kdb_binary_eval_scan_ratio_missing")
    binary_scan_ratio = float(binary_scan_ratio_match.group(1))
    require(
        kdb_binary_total >= KDB_BINARY_MIN_CASES and kdb_binary_passed == kdb_binary_total,
        f"kdb_binary_eval_pass_rate={kdb_binary_passed}/{kdb_binary_total}",
    )
    require(binary_scan_ratio <= KDB_BINARY_MAX_SCAN_RATIO, f"kdb_binary_scan_ratio={binary_scan_ratio:.3f}")
    kdb_term_text = read(kdb_term_index_report)
    require("Status: `PASS`" in kdb_term_text, "kdb_term_index_eval_not_pass")
    kdb_term_match = re.search(r"Term-index recall pass rate:\s+`(\d+)/(\d+)`", kdb_term_text)
    require(kdb_term_match is not None, "kdb_term_index_eval_pass_rate_missing")
    kdb_term_passed = int(kdb_term_match.group(1))
    kdb_term_total = int(kdb_term_match.group(2))
    term_ratio_match = re.search(r"Candidate row ratio:\s+`([0-9.]+)`", kdb_term_text)
    require(term_ratio_match is not None, "kdb_term_index_eval_row_ratio_missing")
    term_ratio = float(term_ratio_match.group(1))
    require(
        kdb_term_total >= KDB_TERM_INDEX_MIN_CASES and kdb_term_passed == kdb_term_total,
        f"kdb_term_index_eval_pass_rate={kdb_term_passed}/{kdb_term_total}",
    )
    require(term_ratio <= KDB_TERM_INDEX_MAX_ROW_RATIO, f"kdb_term_index_row_ratio={term_ratio:.3f}")
    for pack_id in ("CHAT", "DOSHELP", "OFFICE"):
        pack = pack_by_id[pack_id]
        report = read(evidence_dir / f"quality_report_assistant_{pack.pack_id.lower()}.md")
        match = re.search(r"Prompt pass rate:\s+`(\d+)/(\d+)`", report)
        require(match is not None, f"pack_quality_pass_rate_missing={pack.pack_id}")
        passed = int(match.group(1))
        total = int(match.group(2))
        if pack_id == "CHAT":
            require(total > 0 and passed / total >= 0.75, f"pack_quality_pass_rate={pack.pack_id}:{passed}/{total}")
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
    require("DEV pack" in assist, "dev_usage_missing")
    require("PORTABLE pack" in assist, "portable_usage_missing")
    require("You are a small friendly DOS conversation assistant. User:" not in assist, "chat_prompt_echo_in_log")
    require("ASSIST_REPLY|pack=CHAT|intent=general_chat" in assist, "chat_reply_missing")
    require("ASSIST_REPLY|pack=DOSHELP|intent=dos_memory" in assist, "doshelp_reply_missing")
    require("ASSIST_REPLY|pack=OFFICE|intent=office_rewrite" in assist, "office_reply_missing")
    require("ASSIST_REPLY|pack=DEV|intent=general_chat" in assist, "dev_reply_missing")
    require("ASSIST_REPLY|pack=PORTABLE|intent=general_chat" in assist, "portable_reply_missing")
    require("status=model_unavailable" not in assist, "model_unavailable_in_assistant_log")


def verify_stress_logs(stress_log: Path, stress_compile_log: Path, stress_report: Path) -> None:
    stress = read(stress_log)
    compile_text = read(stress_compile_log)
    report = read(stress_report)
    require("ASSIST_COMPILE_OK" in compile_text, "stress_compile_marker_missing")
    require("ASSIST_BEGIN|suite=stress-probe|version=1" in stress, "stress_begin_marker_missing")
    require("ASSIST_END|suite=stress-probe|packs=5" in stress, "stress_end_marker_missing")
    require(stress.count("ASSIST_REPLY|") == STRESS_REPLY_COUNT, "stress_reply_count_mismatch")
    require("status=model_unavailable" not in stress, "model_unavailable_in_stress_log")
    require("|query=" in stress and "|answer=" in stress, "stress_structured_answer_missing")
    require("|source=memory|" in stress, "stress_memory_source_missing")
    require("Status: `PASS`" in report, "stress_report_not_pass")
    require(f"Reply count: `{STRESS_REPLY_COUNT}`" in report, "stress_report_reply_count")
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
    parser.add_argument("--generalist-prompt-report", type=Path, default=DEFAULT_GENERALIST_PROMPT_REPORT)
    parser.add_argument("--retrieval-report", type=Path, default=DEFAULT_RETRIEVAL_REPORT)
    parser.add_argument("--usefulness-report", type=Path, default=DEFAULT_USEFULNESS_REPORT)
    parser.add_argument("--kdb-index-report", type=Path, default=DEFAULT_KDB_INDEX_REPORT)
    parser.add_argument("--kdb-binary-report", type=Path, default=DEFAULT_KDB_BINARY_REPORT)
    parser.add_argument("--kdb-term-index-report", type=Path, default=DEFAULT_KDB_TERM_INDEX_REPORT)
    args = parser.parse_args()

    packs = verify_pack_files(args.pack_root)
    verify_source()
    verify_pack_quality(
        packs,
        args.evidence_dir,
        args.raw_prompt_report,
        args.generalist_prompt_report,
        args.retrieval_report,
        args.usefulness_report,
        args.kdb_index_report,
        args.kdb_binary_report,
        args.kdb_term_index_report,
    )
    verify_qemu_logs(packs, args.assistant_log, args.compile_log)
    verify_stress_logs(args.stress_log, args.stress_compile_log, args.stress_report)
    print(f"PROBE_OK assistant_pack_count={len(packs)}")
    print("PROBE_OK assistant_pack_loader=1")
    print("PROBE_OK assistant_pack_models=1")
    print("PROBE_OK assistant_pack_quality=1")
    print("PROBE_OK assistant_generalist_prompt_eval=1")
    print("PROBE_OK assistant_pack_retrieval_eval=1")
    print("PROBE_OK assistant_usefulness_eval=1")
    print("PROBE_OK assistant_kdb_index_eval=1")
    print("PROBE_OK assistant_kdb_binary_eval=1")
    print("PROBE_OK assistant_kdb_term_index_eval=1")
    print("PROBE_OK assistant_model_switch=1")
    print("PROBE_OK assistant_structured_reply=1")
    print("PROBE_OK assistant_art_slots=1")
    print("PROBE_OK assistant_qemu_evidence=1")
    print("PROBE_OK assistant_stress_evidence=1")


if __name__ == "__main__":
    main()

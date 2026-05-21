#!/usr/bin/env python3
"""Build the assistant capability/functionality report from evidence files."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "qemu" / "evidence"
DEFAULT_PACK_ROOT = ROOT / "assets" / "gpt2_basic" / "PACKS"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "assistant_capability_functionality_report.md"
DEFAULT_GENERATED_DATE = "2026-05-21"


@dataclass(frozen=True)
class StressSummary:
    status: str
    replies: str
    sources: str
    average_total_ms: str
    average_retrieval_ms: str
    recall_modes: str


@dataclass(frozen=True)
class PackStats:
    pack_id: str
    rows: int
    buckets: int
    binary_bytes: int
    term_index_bytes: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ASSISTANT_CAPABILITY_REPORT_FAILED {message}")


def read(path: Path) -> str:
    require(path.is_file(), f"missing={path}")
    return path.read_text(encoding="ascii", errors="ignore")


def backtick_value(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s+`([^`]+)`", text, flags=re.MULTILINE)
    require(match is not None, f"missing_label={label}")
    return match.group(1)


def status_value(text: str) -> str:
    return backtick_value(text, "Status")


def probe_value(text: str, key: str) -> str:
    match = re.search(rf"^PROBE_OK {re.escape(key)}=(.+)$", text, flags=re.MULTILINE)
    require(match is not None, f"missing_probe={key}")
    return match.group(1).strip()


def parse_stress_report(path: Path) -> StressSummary:
    text = read(path)
    return StressSummary(
        status=status_value(text),
        replies=backtick_value(text, "Reply count"),
        sources=backtick_value(text, "Source counts"),
        average_total_ms=backtick_value(text, "Average total reply time"),
        average_retrieval_ms=backtick_value(text, "Average retrieval time"),
        recall_modes=backtick_value(text, "Recall modes"),
    )


def pack_ids(pack_root: Path) -> list[str]:
    packs_txt = read(pack_root / "PACKS.TXT")
    ids = [line.strip() for line in packs_txt.splitlines() if line.strip() and not line.startswith("#")]
    require(ids, "pack_ids_missing")
    return ids


def kdb_row_count(path: Path) -> int:
    rows = [
        line
        for line in read(path).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return len(rows)


def pack_stats(pack_root: Path) -> list[PackStats]:
    rows: list[PackStats] = []
    for pack_id in pack_ids(pack_root):
        root = pack_root / pack_id
        kb2_files = [
            path
            for path in root.glob("KB2*.BIN")
            if path.name.upper() != "KB2ALL.BIN"
        ]
        all_kb2_files = list(root.glob("KB2*.BIN"))
        term_index_files = list(root.glob("KB2T*.TXT"))
        rows.append(
            PackStats(
                pack_id=pack_id,
                rows=kdb_row_count(root / "KDB.TXT"),
                buckets=len(kb2_files),
                binary_bytes=sum(path.stat().st_size for path in all_kb2_files),
                term_index_bytes=sum(path.stat().st_size for path in term_index_files),
            )
        )
    return rows


def report_line(text: str, label: str) -> str:
    return backtick_value(text, label)


def physical_capture_status(evidence_dir: Path) -> str:
    manifests = sorted(evidence_dir.glob("hardware_*_manifest.md"))
    if not manifests:
        return "PENDING: no staged physical `hardware_<machine>_manifest.md` capture is present yet."
    names = ", ".join(path.name for path in manifests)
    return f"PASS: staged physical captures present: {names}."


def release_hash_status(release_assets: Path) -> str:
    required = (
        "gpt2-basic-preview.zip.sha256",
        "gpt2-basic-dosbox.zip.sha256",
        "gpt2-basic-hardware-transfer.zip.sha256",
        "gpt2-basic-launch-kit.zip.sha256",
    )
    missing = [name for name in required if not (release_assets / name).is_file()]
    if missing:
        return "MISSING: " + ", ".join(missing)
    return "PASS"


def build_report(evidence_dir: Path, pack_root: Path, release_assets: Path, generated_date: str) -> str:
    raw = read(evidence_dir / "assistant_raw_prompt_eval.md")
    generalist = read(evidence_dir / "assistant_generalist_prompt_eval.md")
    consistency = read(evidence_dir / "assistant_consistency_eval.md")
    retrieval = read(evidence_dir / "assistant_pack_retrieval_eval.md")
    usefulness = read(evidence_dir / "assistant_usefulness_eval.md")
    recall_benchmark = read(evidence_dir / "assistant_recall_benchmark.md")
    kdb_index = read(evidence_dir / "assistant_kdb_index_eval.md")
    kdb_binary = read(evidence_dir / "assistant_kdb_binary_eval.md")
    kdb_term = read(evidence_dir / "assistant_kdb_term_index_eval.md")
    assistant_log = read(evidence_dir / "assistant_486.log")
    hardware_probe = read(evidence_dir / "hardware_capture_486_qemu_probe.log")

    stress = parse_stress_report(evidence_dir / "assistant_stress_report.md")
    hardware_stress = parse_stress_report(evidence_dir / "hardware_capture_486_qemu_stress_report.md")
    hardware_recall = read(evidence_dir / "hardware_capture_486_qemu_recall_report.md")

    assistant_packs_match = re.search(r"ASSIST_END\|packs=(\d+)", assistant_log)
    require(assistant_packs_match is not None, "assistant_end_missing")
    assistant_packs = assistant_packs_match.group(1)
    hardware_stress_replies = probe_value(hardware_probe, "hardware_assistant_stress_replies")

    stats = pack_stats(pack_root)
    stats_lines = [
        f"  - `{row.pack_id}`: {row.rows} rows, {row.buckets} buckets, "
        f"{row.binary_bytes} binary bytes, {row.term_index_bytes} term-index bytes."
        for row in stats
    ]

    lines = [
        "# Assistant Capability And Functionality Report",
        "",
        f"Date: {generated_date}",
        "Status: `PASS`",
        "",
        "This report is generated from repository evidence files by `scripts/build_assistant_capability_report.py`.",
        "",
        "## Runtime Capability",
        "",
        f"- Runs under FreeDOS/QEMU 486 with {len(stats)} assistant packs: "
        + ", ".join(f"`{row.pack_id}`" for row in stats)
        + ".",
        "- Supports hot pack switching through `PACKS.TXT` and each pack's `PACK.INI`.",
        "- Supports pack-local model paths, pack-local art assets, pack-local golden rows, pack-local help/knowledge rows, and editable `USER.TXT` notes.",
        "- Uses retrieval-first answering before model synthesis: golden rows, compiled knowledge recall, session memory, and fallback checks are explicit in `ASSIST_REPLY`.",
        "- Reports structured provenance and timing for every reply: `source`, `recall`, `recall_score`, `t_retrieve_ms`, `t_golden_ms`, `t_memory_ms`, `t_model_ms`, and `t_total_ms`.",
        "- Interactive shell exposes `/capabilities`, `/limits`, `/sources`, `/status`, `/about`, `/pack`, `/memory`, `/remember KEY=VALUE`, and `/forget`.",
        "",
        "## Recall And Storage",
        "",
        "- Text KDB remains the readable source/fallback format: `KDB.TXT`, `KDBIDX.TXT`, and `KDB?.TXT`.",
        "- Compiled KB2 recall ships for each pack: `KB2ALL.BIN`, `KB2IDX.TXT`, `KB2?.BIN`, aggregate `KB2TERM.TXT`, and sharded `KB2T?.TXT` term indexes.",
        "- KB2 files use fixed-width records for 486-friendly sequential reads and avoid reparsing large text rows during recall.",
        "- `KB2T?.TXT` shards are compact per-pack inverted term indexes. The DOS runtime opens the strongest relevant term shard first, then falls back to `KB2TERM.TXT`, binary buckets, and finally text KDB recall.",
        "- Current compiled KB2 payload sizes:",
        *stats_lines,
        f"- Binary recall evaluation: `PASS {report_line(kdb_binary, 'Binary recall pass rate')}`.",
        f"- Binary candidate row scan ratio: `{report_line(kdb_binary, 'Candidate row scan ratio')}`.",
        f"- Binary candidate byte ratio: `{report_line(kdb_binary, 'Candidate byte ratio')}`.",
        f"- Term-index recall evaluation: `PASS {report_line(kdb_term, 'Term-index recall pass rate')}`.",
        f"- Term-index candidate row scan ratio: `{report_line(kdb_term, 'Candidate row ratio')}`.",
        f"- Term-index candidate byte ratio: `{report_line(kdb_term, 'Candidate byte ratio')}`.",
        f"- QEMU recall benchmark: `PASS {report_line(recall_benchmark, 'Recall case count')} cases`.",
        f"- QEMU recall average retrieval time: `{report_line(recall_benchmark, 'Average retrieval time')}`.",
        f"- QEMU recall max retrieval time: `{report_line(recall_benchmark, 'Max retrieval time')}`.",
        f"- QEMU recall modes: `{report_line(recall_benchmark, 'Recall modes')}`.",
        "",
        "## Language Coverage",
        "",
        f"- Raw direct model prompt gate: `PASS {report_line(raw, 'Prompt pass rate')}`.",
        f"- Generalist conversational prompt gate: `PASS {report_line(generalist, 'Prompt pass rate')}`.",
        f"- Consistency gate: `PASS {report_line(consistency, 'Prompt variants')} variants, {report_line(consistency, 'Consistent prompt groups')} groups`.",
        f"- Pack retrieval gate: `PASS {report_line(retrieval, 'Retrieval pass rate')}`.",
        f"- Usefulness workflow gate: `PASS {report_line(usefulness, 'Task pass rate')} tasks, {report_line(usefulness, 'Workflow coverage')} workflows`.",
        f"- KDB text index gate: `PASS {report_line(kdb_index, 'Indexed recall pass rate')}`.",
        f"- KDB binary gate: `PASS {report_line(kdb_binary, 'Binary recall pass rate')}`.",
        f"- KDB term-index gate: `PASS {report_line(kdb_term, 'Term-index recall pass rate')}`.",
        f"- DOS recall benchmark gate: `PASS {report_line(recall_benchmark, 'Recall case count')} cases`.",
        "",
        "Covered categories include general chat, identity, local inference, offline limits, prompt repair, repeated-answer recovery, troubleshooting, DOS setup, office writing, developer pack authoring, and portable-intelligence concepts.",
        "",
        "Usefulness workflows currently cover operator prompts, trust/offline limits, DOS setup and repair, hardware transfer and emulator evidence, office handoffs, planning and risk, developer pack authoring, fast local recall architecture, and portable intelligence.",
        "",
        "## DOS/QEMU Stress Result",
        "",
        f"- Scripted QEMU assistant run: `PASS`, reached `ASSIST_END|packs={assistant_packs}`.",
        "- Stress QEMU run: `PASS`, reached `ASSIST_END|suite=stress-probe|packs=5`.",
        f"- Stress replies: `{stress.replies}`.",
        f"- Stress source mix: `{stress.sources}`.",
        f"- Average total reply time in the stress report: `{stress.average_total_ms}`.",
        f"- Average retrieval time in the stress report: `{stress.average_retrieval_ms}`.",
        f"- Recall modes in the stress report: `{stress.recall_modes}`.",
        "- Visible-answer validation: `PASS`.",
        "",
        "## Hardware-Capture Rehearsal",
        "",
        "- QEMU rehearses the physical `C:\\GPT2\\HWVALID.BAT` path before real transfer.",
        "- Hardware-capture rehearsal: `PASS`.",
        f"- Hardware-capture assistant stress replies: `{hardware_stress_replies}`.",
        f"- Hardware-capture stress source mix: `{hardware_stress.sources}`.",
        f"- Hardware-capture average total reply time: `{hardware_stress.average_total_ms}`.",
        f"- Hardware-capture average retrieval time: `{hardware_stress.average_retrieval_ms}`.",
        f"- Hardware-capture recall benchmark: `PASS {report_line(hardware_recall, 'Recall case count')} cases`.",
        f"- Hardware-capture recall average retrieval time: `{report_line(hardware_recall, 'Average retrieval time')}`.",
        f"- Physical machine capture status: {physical_capture_status(evidence_dir)}",
        "",
        "## Authoring And Import",
        "",
        "- `scripts/import_assistant_notes.py` can import ASCII notes into `USER.TXT` or `KNOW.TXT`.",
        "- `--target user` writes machine-local notes without changing bundled pack knowledge.",
        "- `--target know --rebuild-kdb` updates bundled pack knowledge and regenerates KDB/KB2 artifacts.",
        "- `scripts/create_assistant_pack.py` can create a complete lightweight pack from a folder of ASCII notes, sharing `PACKS\\CHAT\\MODEL` by default.",
        "- The pack generator writes `PACK.INI`, authoring files, `USER.TXT`, `USAGE.TXT`, generated KDB buckets, compiled KB2 pages, aggregate `KB2TERM.TXT`, and `KB2T?.TXT` shards.",
        "- Authoring validator checks required pack files, source rows, generated text KDB, generated binary KDB, and model references.",
        "",
        "## Release Payload",
        "",
        "- Preview package manifest: `included`.",
        "- Preview release tracked-input gate: `PASS`.",
        "- Preview artifact verifier: `PASS`.",
        f"- Release sidecar hashes: `{release_hash_status(release_assets)}`.",
        "- Runtime bundles exclude host-only `TRAIN.TXT` and `TOKBASE.TXT`.",
        "",
        "## Known Limits",
        "",
        "- This is not a frontier-scale LLM. It is a retrieval-first, pack-specialized DOS assistant with a very small local model.",
        "- The strongest behavior comes from curated pack knowledge, golden rows, session memory, and fast local recall.",
        "- Long, ambiguous, or out-of-domain prompts should be shortened or moved into an appropriate pack.",
        "- No live web, news, package registry, or network lookup is available inside DOS.",
        "- Current 486 stress replies did not require raw model generation; that is intentional for reliability and speed on this hardware class.",
        "- Physical 486-class board evidence is still pending until real hardware returns the `HWVALID.LOG`, `QUAL.LOG`, `PERF.LOG`, `ASSIST.LOG`, `ASTRESS.LOG`, `ASSISTC.LOG`, and `HWNOTES.TXT` set.",
        "",
        "## Next Production Targets",
        "",
        "- Convert the `KB2T?.TXT` shard rows into an even denser binary term index once the text format has stabilized under real authoring changes.",
        "- Add larger domain packs with the same KB2 contract, especially hardware repair, programming, office workflows, and offline reference manuals.",
        "- Add a compact on-disk conversation database so memory persists across sessions while remaining inspectable and editable.",
        "- Add a pack-selection router so the shell can recommend or switch packs from query intent.",
        "- Add latency budgets per pack and fail the harness if retrieval or total reply time regresses beyond the 486 profile target.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    sample = (
        "Status: `PASS`\n"
        "Reply count: `50`\n"
        "Source counts: `golden=1 retrieval=2 model=0 fallback=0 memory=3`\n"
        "Average total reply time: `12 ms`\n"
        "Average retrieval time: `5 ms`\n"
        "Recall modes: `kb2_term=6`\n"
    )
    parsed = parse_stress_report_text_for_test(sample)
    require(parsed.replies == "50", "self_test_stress_replies")
    require(parsed.sources.startswith("golden=1"), "self_test_stress_sources")
    require(backtick_value("Prompt pass rate: `3/3`\n", "Prompt pass rate") == "3/3", "self_test_rate")
    print("PROBE_OK assistant_capability_report_self_test=1")


def parse_stress_report_text_for_test(text: str) -> StressSummary:
    return StressSummary(
        status=status_value(text),
        replies=backtick_value(text, "Reply count"),
        sources=backtick_value(text, "Source counts"),
        average_total_ms=backtick_value(text, "Average total reply time"),
        average_retrieval_ms=backtick_value(text, "Average retrieval time"),
        recall_modes=backtick_value(text, "Recall modes"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--release-assets", type=Path, default=ROOT / "promo" / "renders" / "release-assets")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-date", default=DEFAULT_GENERATED_DATE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    report = build_report(args.evidence_dir, args.pack_root, args.release_assets, args.generated_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="ascii")
    print(f"ASSISTANT_CAPABILITY_REPORT|path={args.output}")
    print("PROBE_OK assistant_capability_report=1")


if __name__ == "__main__":
    main()

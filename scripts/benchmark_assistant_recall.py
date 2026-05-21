#!/usr/bin/env python3
"""Benchmark ASSIST.EXE recall-probe logs for coverage and latency."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_assistant_pack_retrieval import CASES, RetrievalCase, validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "qemu" / "evidence" / "assistant_recall_486.log"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_recall_benchmark.md"
DEFAULT_MAX_AVERAGE_MS = 250
DEFAULT_MAX_SINGLE_MS = 1500


@dataclass(frozen=True)
class RecallRecord:
    pack: str
    query: str
    recall: str
    score: int
    retrieve_ms: int
    answer: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ASSISTANT_RECALL_BENCHMARK_FAILED {message}")


def parse_record(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in line.rstrip().split("|")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def parse_records(text: str) -> list[RecallRecord]:
    records: list[RecallRecord] = []
    for line in text.splitlines():
        if not line.startswith("ASSIST_RECALL|"):
            continue
        fields = parse_record(line)
        score_text = fields.get("recall_score", "")
        timing_text = fields.get("t_retrieve_ms", "")
        require(score_text.isdigit(), f"bad_score={fields.get('pack', '')}:{fields.get('query', '')}")
        require(timing_text.isdigit(), f"bad_timing={fields.get('pack', '')}:{fields.get('query', '')}")
        records.append(
            RecallRecord(
                fields.get("pack", ""),
                fields.get("query", ""),
                fields.get("recall", ""),
                int(score_text),
                int(timing_text),
                fields.get("answer", ""),
            )
        )
    return records


def expected_cases() -> dict[tuple[str, str], RetrievalCase]:
    return {(case.pack, case.query): case for case in CASES}


def validate_records(
    records: list[RecallRecord],
    max_average_ms: int,
    max_single_ms: int,
) -> tuple[Counter[str], Counter[str]]:
    expected = expected_cases()
    seen: set[tuple[str, str]] = set()
    pack_counts: Counter[str] = Counter()
    recall_counts: Counter[str] = Counter()
    require(len(records) == len(expected), f"record_count={len(records)}/{len(expected)}")
    for record in records:
        key = (record.pack, record.query)
        require(key in expected, f"unexpected_recall={record.pack}:{record.query}")
        require(key not in seen, f"duplicate_recall={record.pack}:{record.query}")
        seen.add(key)
        pack_counts[record.pack] += 1
        recall_counts[record.recall] += 1
        require(record.recall not in ("", "none"), f"no_recall_mode={record.pack}:{record.query}")
        require(record.score > 0, f"nonpositive_score={record.pack}:{record.query}")
        require(record.answer.strip(), f"empty_answer={record.pack}:{record.query}")
        require(record.retrieve_ms <= max_single_ms, f"slow_recall={record.pack}:{record.query}:{record.retrieve_ms}")
        reason = validate(expected[key], record.answer)
        require(reason is None, f"{reason}:{record.pack}:{record.query}:{record.answer}")
    missing = sorted(set(expected) - seen)
    require(not missing, f"missing_recall={missing}")
    average_ms = sum(record.retrieve_ms for record in records) // len(records)
    require(average_ms <= max_average_ms, f"average_recall_ms={average_ms}>{max_average_ms}")
    require(recall_counts["kb2_term"] + recall_counts["kb2_bucket"] + recall_counts["kb2_full"] >= len(records) // 2, "too_little_binary_recall")
    return pack_counts, recall_counts


def markdown_report(records: list[RecallRecord], pack_counts: Counter[str], recall_counts: Counter[str]) -> str:
    timings = [record.retrieve_ms for record in records]
    scores = [record.score for record in records]
    average_ms = sum(timings) // len(timings) if timings else 0
    max_ms = max(timings) if timings else 0
    average_score = sum(scores) // len(scores) if scores else 0
    lines = [
        "# Assistant Recall Benchmark",
        "",
        "Status: `PASS`",
        f"Recall case count: `{len(records)}`",
        f"Average retrieval time: `{average_ms} ms`",
        f"Max retrieval time: `{max_ms} ms`",
        f"Average recall score: `{average_score}`",
        "Pack counts: `" + " ".join(f"{pack}={count}" for pack, count in sorted(pack_counts.items())) + "`",
        "Recall modes: `" + " ".join(f"{mode}={count}" for mode, count in sorted(recall_counts.items())) + "`",
        "",
        "This benchmark is generated from `ASSIST.EXE --recall-probe` and measures local pack recall without model generation.",
        "",
        "| Pack | Recall | Score | Retrieve ms | Query | Answer |",
        "|---|---|---:|---:|---|---|",
    ]
    for record in records:
        lines.append(
            "| {pack} | {recall} | {score} | {ms} | {query} | {answer} |".format(
                pack=record.pack,
                recall=record.recall,
                score=record.score,
                ms=record.retrieve_ms,
                query=record.query.replace("|", "/"),
                answer=record.answer.replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_benchmark(log: Path, report: Path, max_average_ms: int, max_single_ms: int) -> int:
    text = log.read_text(encoding="ascii", errors="ignore")
    require("ASSIST_BEGIN|suite=recall-probe|version=1" in text, "recall_begin_missing")
    require("ASSIST_END|suite=recall-probe|packs=5" in text, "recall_end_missing")
    records = parse_records(text)
    pack_counts, recall_counts = validate_records(records, max_average_ms, max_single_ms)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(records, pack_counts, recall_counts), encoding="ascii")
    average_ms = sum(record.retrieve_ms for record in records) // len(records)
    max_ms = max(record.retrieve_ms for record in records)
    print(f"PROBE_OK assistant_recall_benchmark_cases={len(records)}")
    print(
        "ASSISTANT_RECALL_BENCHMARK|"
        f"cases={len(records)}|average_ms={average_ms}|max_ms={max_ms}|report={report}"
    )
    print("PROBE_OK assistant_recall_benchmark_pass=1")
    return 0


def self_test() -> None:
    lines = ["ASSIST_BEGIN|suite=recall-probe|version=1"]
    for case in CASES:
        answer = " ".join(case.terms) + "."
        lines.append(
            "ASSIST_RECALL|pack={pack}|query={query}|recall=kb2_term|"
            "recall_score=99|t_retrieve_ms=3|answer={answer}".format(
                pack=case.pack,
                query=case.query,
                answer=answer,
            )
        )
    lines.append("ASSIST_END|suite=recall-probe|packs=5")
    records = parse_records("\n".join(lines))
    pack_counts, recall_counts = validate_records(records, DEFAULT_MAX_AVERAGE_MS, DEFAULT_MAX_SINGLE_MS)
    report = markdown_report(records, pack_counts, recall_counts)
    assert "Status: `PASS`" in report
    assert f"Recall case count: `{len(CASES)}`" in report
    print("PROBE_OK assistant_recall_benchmark_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-average-ms", type=int, default=DEFAULT_MAX_AVERAGE_MS)
    parser.add_argument("--max-single-ms", type=int, default=DEFAULT_MAX_SINGLE_MS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    raise SystemExit(run_benchmark(args.log, args.report, args.max_average_ms, args.max_single_ms))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate compiled binary assistant KDB recall coverage and scan reduction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from assistant_pack_contract import HelpRow, PackContract, load_all_pack_contracts
from build_assistant_kdb import (
    KDB2_BODY_BYTES,
    KDB2_HEADER_BYTES,
    KDB2_RECORD_BYTES,
    KDB2_TERMS_BYTES,
    KDB2_TITLE_BYTES,
    kdb2_bucket_name,
)
from evaluate_assistant_kdb_index import query_buckets
from evaluate_assistant_pack_retrieval import CASES, RetrievalCase, retrieval_score, validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_kdb_binary_eval.md"


@dataclass(frozen=True)
class KdbBinaryResult:
    case: RetrievalCase
    buckets: tuple[str, ...]
    full_rows: int
    candidate_rows: int
    candidate_bytes: int
    full_bytes: int
    title: str
    answer: str
    score: int
    reason: str | None


def _decode_field(payload: bytes) -> str:
    return payload.decode("ascii").strip()


def read_kdb2_rows(path: Path) -> list[HelpRow]:
    payload = path.read_bytes()
    if len(payload) < KDB2_HEADER_BYTES:
        raise ValueError(f"KDB2 file too short: {path}")
    if not payload.startswith(b"KDB2|V=1|"):
        raise ValueError(f"KDB2 file has invalid header: {path}")
    body = payload[KDB2_HEADER_BYTES:]
    if len(body) % KDB2_RECORD_BYTES != 0:
        raise ValueError(f"KDB2 file has partial record: {path}")
    rows: list[HelpRow] = []
    offset = 0
    while offset < len(body):
        terms = _decode_field(body[offset : offset + KDB2_TERMS_BYTES])
        offset += KDB2_TERMS_BYTES
        title = _decode_field(body[offset : offset + KDB2_TITLE_BYTES])
        offset += KDB2_TITLE_BYTES
        text = _decode_field(body[offset : offset + KDB2_BODY_BYTES])
        offset += KDB2_BODY_BYTES
        rows.append(HelpRow(terms, title, text))
    return rows


def binary_candidate_rows(pack: PackContract, buckets: tuple[str, ...]) -> tuple[list[HelpRow], int]:
    rows: list[HelpRow] = []
    seen: set[tuple[str, str]] = set()
    scanned_bytes = 0
    for bucket in buckets:
        bucket_path = pack.kdb_bin_path.parent / kdb2_bucket_name(bucket)
        if not bucket_path.exists():
            continue
        scanned_bytes += bucket_path.stat().st_size
        for row in read_kdb2_rows(bucket_path):
            identity = (row.title, row.text)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)
    return rows, scanned_bytes


def retrieve_from_binary(pack: PackContract, case: RetrievalCase) -> KdbBinaryResult:
    buckets = query_buckets(case.query)
    full_rows = read_kdb2_rows(pack.kdb_bin_path)
    candidates, candidate_bytes = binary_candidate_rows(pack, buckets)
    best_row: HelpRow | None = None
    best_score = 0
    for row in candidates:
        score = retrieval_score(case.query, row)
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return KdbBinaryResult(
            case,
            buckets,
            len(full_rows),
            0,
            candidate_bytes,
            pack.kdb_bin_path.stat().st_size,
            "",
            "",
            0,
            "no_binary_bucket_match",
        )
    answer = f"{best_row.title}: {best_row.text}"
    return KdbBinaryResult(
        case,
        buckets,
        len(full_rows),
        len(candidates),
        candidate_bytes,
        pack.kdb_bin_path.stat().st_size,
        best_row.title,
        best_row.text,
        best_score,
        validate(case, answer),
    )


def markdown_report(results: list[KdbBinaryResult]) -> str:
    passed = sum(1 for result in results if result.reason is None)
    full_rows = sum(result.full_rows for result in results)
    candidate_rows_total = sum(result.candidate_rows for result in results)
    row_ratio = candidate_rows_total / full_rows if full_rows else 1.0
    full_bytes = sum(result.full_bytes for result in results)
    candidate_bytes = sum(result.candidate_bytes for result in results)
    byte_ratio = candidate_bytes / full_bytes if full_bytes else 1.0
    status = "PASS" if passed == len(results) else "FAIL"
    lines = [
        "# Assistant KDB Binary Evaluation",
        "",
        f"Status: `{status}`",
        f"Binary recall pass rate: `{passed}/{len(results)}`",
        f"Candidate rows scanned: `{candidate_rows_total}/{full_rows}`",
        f"Candidate row scan ratio: `{row_ratio:.3f}`",
        f"Candidate bytes opened: `{candidate_bytes}/{full_bytes}`",
        f"Candidate byte ratio: `{byte_ratio:.3f}`",
        "",
        "This gate mirrors the DOS KB2*.BIN fast path before text KDB fallback.",
        "",
        "| Pack | Query | Buckets | Rows | Bytes | Status | Reason | Answer |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| {pack} | {query} | {buckets} | {rows} | {bytes_} | {status} | {reason} | {answer} |".format(
                pack=result.case.pack,
                query=result.case.query.replace("|", "/"),
                buckets=",".join(result.buckets),
                rows=f"{result.candidate_rows}/{result.full_rows}",
                bytes_=f"{result.candidate_bytes}/{result.full_bytes}",
                status="PASS" if result.reason is None else "FAIL",
                reason=result.reason or "",
                answer=(result.title + ": " + result.answer).replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(report: Path, max_row_scan_ratio: float) -> int:
    packs = {pack.pack_id: pack for pack in load_all_pack_contracts()}
    results = [retrieve_from_binary(packs[case.pack], case) for case in CASES]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(results), encoding="ascii")
    passed = sum(1 for result in results if result.reason is None)
    full_rows = sum(result.full_rows for result in results)
    candidate_rows_total = sum(result.candidate_rows for result in results)
    ratio = candidate_rows_total / full_rows if full_rows else 1.0
    print(f"PROBE_OK assistant_kdb_binary_eval_cases={len(results)}")
    print(
        "ASSISTANT_KDB_BINARY_EVAL|"
        f"pass={passed}|total={len(results)}|row_scan_ratio={ratio:.3f}|report={report}"
    )
    if passed != len(results) or ratio > max_row_scan_ratio:
        print(
            "ASSISTANT_KDB_BINARY_EVAL_FAILED|"
            f"pass={passed}|total={len(results)}|row_scan_ratio={ratio:.3f}"
        )
        return 1
    print("PROBE_OK assistant_kdb_binary_eval_pass=1")
    return 0


def self_test() -> None:
    pack = {pack.pack_id: pack for pack in load_all_pack_contracts()}["DEV"]
    rows = read_kdb2_rows(pack.kdb_bin_path)
    assert rows
    assert rows[0].title
    case = next(case for case in CASES if case.pack == "DEV" and case.query == "how can this feel modern on a 486")
    result = retrieve_from_binary(pack, case)
    assert result.reason is None
    assert result.candidate_rows < result.full_rows
    report = markdown_report([result])
    assert "Status: `PASS`" in report
    print("PROBE_OK assistant_kdb_binary_eval_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-row-scan-ratio", type=float, default=0.65)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    raise SystemExit(run_eval(args.report, args.max_row_scan_ratio))


if __name__ == "__main__":
    main()

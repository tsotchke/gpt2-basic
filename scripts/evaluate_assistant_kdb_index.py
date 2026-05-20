#!/usr/bin/env python3
"""Evaluate bucketed assistant KDB recall coverage and scan reduction."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from assistant_pack_contract import HelpRow, PackContract, load_all_pack_contracts
from build_assistant_kdb import STOPWORDS, bucketed_rows
from evaluate_assistant_pack_retrieval import CASES, RetrievalCase, retrieval_score, validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_kdb_index_eval.md"


@dataclass(frozen=True)
class KdbIndexResult:
    case: RetrievalCase
    buckets: tuple[str, ...]
    full_rows: int
    candidate_rows: int
    title: str
    answer: str
    score: int
    reason: str | None


def query_buckets(query: str) -> tuple[str, ...]:
    buckets: list[str] = []
    for word in re.findall(r"[a-z0-9]+", query.lower()):
        if len(word) < 3 or word in STOPWORDS:
            continue
        bucket = word[0].upper()
        if bucket not in buckets:
            buckets.append(bucket)
    return tuple(buckets)


def candidate_rows(pack: PackContract, buckets: tuple[str, ...]) -> list[HelpRow]:
    rows_by_bucket = bucketed_rows(pack)
    rows: list[HelpRow] = []
    seen: set[tuple[str, str]] = set()
    for bucket in buckets:
        for row in rows_by_bucket.get(bucket, []):
            identity = (row.title, row.body)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(HelpRow(row.terms, row.title, row.body))
    return rows


def retrieve_from_index(pack: PackContract, case: RetrievalCase) -> KdbIndexResult:
    buckets = query_buckets(case.query)
    candidates = candidate_rows(pack, buckets)
    best_row: HelpRow | None = None
    best_score = 0
    for row in candidates:
        score = retrieval_score(case.query, row)
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return KdbIndexResult(case, buckets, len(pack.kdb_rows), 0, "", "", 0, "no_bucket_match")
    answer = f"{best_row.title}: {best_row.text}"
    return KdbIndexResult(
        case,
        buckets,
        len(pack.kdb_rows),
        len(candidates),
        best_row.title,
        best_row.text,
        best_score,
        validate(case, answer),
    )


def markdown_report(results: list[KdbIndexResult]) -> str:
    passed = sum(1 for result in results if result.reason is None)
    full_rows = sum(result.full_rows for result in results)
    candidate_rows_total = sum(result.candidate_rows for result in results)
    ratio = candidate_rows_total / full_rows if full_rows else 1.0
    status = "PASS" if passed == len(results) else "FAIL"
    lines = [
        "# Assistant KDB Index Evaluation",
        "",
        f"Status: `{status}`",
        f"Indexed recall pass rate: `{passed}/{len(results)}`",
        f"Candidate rows scanned: `{candidate_rows_total}/{full_rows}`",
        f"Candidate scan ratio: `{ratio:.3f}`",
        "",
        "This gate mirrors the DOS bucket fast path before the full KDB fallback.",
        "",
        "| Pack | Query | Buckets | Rows | Status | Reason | Answer |",
        "|---|---|---|---:|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| {pack} | {query} | {buckets} | {rows} | {status} | {reason} | {answer} |".format(
                pack=result.case.pack,
                query=result.case.query.replace("|", "/"),
                buckets=",".join(result.buckets),
                rows=f"{result.candidate_rows}/{result.full_rows}",
                status="PASS" if result.reason is None else "FAIL",
                reason=result.reason or "",
                answer=(result.title + ": " + result.answer).replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(report: Path, max_scan_ratio: float) -> int:
    packs = {pack.pack_id: pack for pack in load_all_pack_contracts()}
    results = [retrieve_from_index(packs[case.pack], case) for case in CASES]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(results), encoding="ascii")
    passed = sum(1 for result in results if result.reason is None)
    full_rows = sum(result.full_rows for result in results)
    candidate_rows_total = sum(result.candidate_rows for result in results)
    ratio = candidate_rows_total / full_rows if full_rows else 1.0
    print(f"PROBE_OK assistant_kdb_index_eval_cases={len(results)}")
    print(
        "ASSISTANT_KDB_INDEX_EVAL|"
        f"pass={passed}|total={len(results)}|scan_ratio={ratio:.3f}|report={report}"
    )
    if passed != len(results) or ratio > max_scan_ratio:
        print(
            "ASSISTANT_KDB_INDEX_EVAL_FAILED|"
            f"pass={passed}|total={len(results)}|scan_ratio={ratio:.3f}"
        )
        return 1
    print("PROBE_OK assistant_kdb_index_eval_pass=1")
    return 0


def self_test() -> None:
    buckets = query_buckets("how can this feel modern on a 486")
    assert buckets == ("F", "M", "4")
    pack = {pack.pack_id: pack for pack in load_all_pack_contracts()}["DEV"]
    result = retrieve_from_index(pack, CASES[-6])
    assert result.reason is None
    assert result.candidate_rows < result.full_rows
    report = markdown_report([result])
    assert "Status: `PASS`" in report
    print("PROBE_OK assistant_kdb_index_eval_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-scan-ratio", type=float, default=0.65)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    raise SystemExit(run_eval(args.report, args.max_scan_ratio))


if __name__ == "__main__":
    main()

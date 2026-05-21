#!/usr/bin/env python3
"""Evaluate KB2 term-index recall coverage and candidate reduction."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from assistant_pack_contract import HelpRow, PackContract, load_all_pack_contracts
from build_assistant_kdb import KDB2_RECORD_BYTES, KDB2_TERM_INDEX_NAME, STOPWORDS
from evaluate_assistant_kdb_binary import read_kdb2_rows
from evaluate_assistant_pack_retrieval import CASES, RetrievalCase, retrieval_score, validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_kdb_term_index_eval.md"


@dataclass(frozen=True)
class KdbTermIndexResult:
    case: RetrievalCase
    terms: tuple[str, ...]
    full_rows: int
    candidate_rows: int
    term_index_bytes: int
    candidate_record_bytes: int
    full_bytes: int
    title: str
    answer: str
    score: int
    reason: str | None


def query_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    for word in re.findall(r"[a-z0-9]+", query.lower()):
        if len(word) < 3 or word in STOPWORDS or word in terms:
            continue
        terms.append(word)
    return tuple(terms)


def read_term_index_bucket(path: Path) -> dict[str, tuple[int, ...]]:
    rows: dict[str, tuple[int, ...]] = {}
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        term, ids = line.split("|", 1)
        rows[term.strip().lower()] = tuple(int(part) for part in ids.split(",") if part.strip())
    return rows


def term_candidate_rows(pack: PackContract, terms: tuple[str, ...]) -> tuple[list[int], int]:
    candidates: list[int] = []
    seen: set[int] = set()
    opened_bytes = 0
    term_set = set(terms)
    index_path = pack.kdb_bin_path.parent / KDB2_TERM_INDEX_NAME
    if not index_path.exists():
        return candidates, opened_bytes
    opened_bytes += index_path.stat().st_size
    for term, row_ids in read_term_index_bucket(index_path).items():
        if term not in term_set:
            continue
        for row_id in row_ids:
            if row_id in seen:
                continue
            seen.add(row_id)
            candidates.append(row_id)
    return candidates, opened_bytes


def retrieve_from_term_index(pack: PackContract, case: RetrievalCase) -> KdbTermIndexResult:
    terms = query_terms(case.query)
    full_rows = read_kdb2_rows(pack.kdb_bin_path)
    candidate_ids, term_index_bytes = term_candidate_rows(pack, terms)
    best_row: HelpRow | None = None
    best_score = 0
    for row_id in candidate_ids:
        if row_id < 1 or row_id > len(full_rows):
            continue
        row = full_rows[row_id - 1]
        score = retrieval_score(case.query, row)
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return KdbTermIndexResult(
            case,
            terms,
            len(full_rows),
            len(candidate_ids),
            term_index_bytes,
            len(candidate_ids) * KDB2_RECORD_BYTES,
            pack.kdb_bin_path.stat().st_size,
            "",
            "",
            0,
            "no_term_index_match",
        )
    answer = f"{best_row.title}: {best_row.text}"
    return KdbTermIndexResult(
        case,
        terms,
        len(full_rows),
        len(candidate_ids),
        term_index_bytes,
        len(candidate_ids) * KDB2_RECORD_BYTES,
        pack.kdb_bin_path.stat().st_size,
        best_row.title,
        best_row.text,
        best_score,
        validate(case, answer),
    )


def markdown_report(results: list[KdbTermIndexResult]) -> str:
    passed = sum(1 for result in results if result.reason is None)
    full_rows = sum(result.full_rows for result in results)
    candidate_rows_total = sum(result.candidate_rows for result in results)
    row_ratio = candidate_rows_total / full_rows if full_rows else 1.0
    full_bytes = sum(result.full_bytes for result in results)
    candidate_bytes = sum(result.term_index_bytes + result.candidate_record_bytes for result in results)
    byte_ratio = candidate_bytes / full_bytes if full_bytes else 1.0
    status = "PASS" if passed == len(results) else "FAIL"
    lines = [
        "# Assistant KDB Term Index Evaluation",
        "",
        f"Status: `{status}`",
        f"Term-index recall pass rate: `{passed}/{len(results)}`",
        f"Candidate rows scored: `{candidate_rows_total}/{full_rows}`",
        f"Candidate row ratio: `{row_ratio:.3f}`",
        f"Term-index plus record bytes touched: `{candidate_bytes}/{full_bytes}`",
        f"Candidate byte ratio: `{byte_ratio:.3f}`",
        "",
        "This gate mirrors the DOS KB2TERM.TXT inverted-index fast path before KB2 bucket fallback.",
        "",
        "| Pack | Query | Terms | Rows | Bytes | Status | Reason | Answer |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for result in results:
        touched = result.term_index_bytes + result.candidate_record_bytes
        lines.append(
            "| {pack} | {query} | {terms} | {rows} | {bytes_} | {status} | {reason} | {answer} |".format(
                pack=result.case.pack,
                query=result.case.query.replace("|", "/"),
                terms=",".join(result.terms),
                rows=f"{result.candidate_rows}/{result.full_rows}",
                bytes_=f"{touched}/{result.full_bytes}",
                status="PASS" if result.reason is None else "FAIL",
                reason=result.reason or "",
                answer=(result.title + ": " + result.answer).replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(report: Path, max_row_ratio: float) -> int:
    packs = {pack.pack_id: pack for pack in load_all_pack_contracts()}
    results = [retrieve_from_term_index(packs[case.pack], case) for case in CASES]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(results), encoding="ascii")
    passed = sum(1 for result in results if result.reason is None)
    full_rows = sum(result.full_rows for result in results)
    candidate_rows_total = sum(result.candidate_rows for result in results)
    ratio = candidate_rows_total / full_rows if full_rows else 1.0
    print(f"PROBE_OK assistant_kdb_term_index_eval_cases={len(results)}")
    print(
        "ASSISTANT_KDB_TERM_INDEX_EVAL|"
        f"pass={passed}|total={len(results)}|row_ratio={ratio:.3f}|report={report}"
    )
    if passed != len(results) or ratio > max_row_ratio:
        print(
            "ASSISTANT_KDB_TERM_INDEX_EVAL_FAILED|"
            f"pass={passed}|total={len(results)}|row_ratio={ratio:.3f}"
        )
        return 1
    print("PROBE_OK assistant_kdb_term_index_eval_pass=1")
    return 0


def self_test() -> None:
    assert query_terms("how can this feel modern on a 486") == ("feel", "modern", "486")
    pack = {pack.pack_id: pack for pack in load_all_pack_contracts()}["DEV"]
    result = retrieve_from_term_index(pack, CASES[-6])
    assert result.reason is None
    assert 0 < result.candidate_rows < result.full_rows
    report = markdown_report([result])
    assert "Status: `PASS`" in report
    print("PROBE_OK assistant_kdb_term_index_eval_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-row-ratio", type=float, default=0.35)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    raise SystemExit(run_eval(args.report, args.max_row_ratio))


if __name__ == "__main__":
    main()

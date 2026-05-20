#!/usr/bin/env python3
"""Evaluate assistant pack HELP/KNOW retrieval on useful paraphrases."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from assistant_pack_contract import HelpRow, PackContract, load_all_pack_contracts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_pack_retrieval_eval.md"
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "you",
    "your",
    "this",
    "that",
    "from",
    "about",
    "into",
    "should",
    "would",
    "could",
}


@dataclass(frozen=True)
class RetrievalCase:
    pack: str
    query: str
    terms: tuple[str, ...]
    min_hits: int = 2


@dataclass(frozen=True)
class RetrievalResult:
    case: RetrievalCase
    source: str
    title: str
    answer: str
    score: int
    reason: str | None


CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase("CHAT", "how can i ask better questions", ("goal", "detail", "next", "step")),
    RetrievalCase("CHAT", "what makes this intelligent on a small computer", ("retrieval", "memory", "model", "tiny"), 3),
    RetrievalCase("CHAT", "which pack should i use for writing", ("chat", "doshelp", "office", "writing"), 3),
    RetrievalCase("CHAT", "can this work without the internet", ("offline", "local", "files", "model"), 3),
    RetrievalCase("CHAT", "how do i recover from a bad answer", ("shorter", "switch", "error", "wrong"), 2),
    RetrievalCase("CHAT", "what proof helps me trust this", ("visible", "local", "weights", "tests", "logs"), 3),
    RetrievalCase("DOSHELP", "what happens before autoexec bat runs", ("config.sys", "autoexec", "drivers", "commands"), 3),
    RetrievalCase("DOSHELP", "why use 8.3 filenames in batches", ("8.3", "dos", "compatibility", "batch"), 3),
    RetrievalCase("DOSHELP", "how should i prepare files for real hardware", ("gpt2", "model", "packs", "cwsdpmi"), 3),
    RetrievalCase("OFFICE", "how should i write a handoff note", ("done", "remains", "evidence", "next"), 3),
    RetrievalCase("OFFICE", "what belongs in a bug report", ("expected", "actual", "steps", "logs"), 3),
    RetrievalCase("OFFICE", "make a compact release note", ("changed", "proof", "limits"), 2),
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieval_score(query: str, row: HelpRow) -> int:
    query_lower = query.lower().strip()
    key = row.key.lower().strip()
    title = row.title.lower().strip()
    body = row.text.lower().strip()
    score = 0
    if key and key in query_lower:
        score += 80 + len(key)
    if len(query_lower) > 3 and query_lower in key:
        score += 40
    if query_lower in title:
        score += 20
    if query_lower in body:
        score += 20
    for word in words(query_lower):
        if len(word) < 3 or word in STOPWORDS:
            continue
        if word in key:
            score += 12
        if word in title:
            score += 6
        if word in body:
            score += 3
    return score


def retrieve(pack: PackContract, query: str) -> tuple[str, HelpRow, int] | None:
    best: tuple[str, HelpRow, int] | None = None
    for source, rows in (("HELP", pack.help_rows), ("KNOW", pack.knowledge_rows)):
        for row in rows:
            score = retrieval_score(query, row)
            if best is None or score > best[2]:
                best = (source, row, score)
    if best is None or best[2] < 8:
        return None
    return best


def validate(case: RetrievalCase, answer: str) -> str | None:
    lower = answer.lower()
    hits = sum(1 for term in case.terms if term.lower() in lower)
    if hits < case.min_hits:
        return f"terms={hits}/{case.min_hits}"
    return None


def markdown_report(results: list[RetrievalResult]) -> str:
    passed = sum(1 for result in results if result.reason is None)
    status = "PASS" if passed == len(results) else "FAIL"
    lines = [
        "# Assistant Pack Retrieval Evaluation",
        "",
        f"Status: `{status}`",
        f"Retrieval pass rate: `{passed}/{len(results)}`",
        "",
        "This gate checks local HELP.TXT and KNOW.TXT retrieval for useful paraphrases before model generation.",
        "",
        "| Pack | Query | Source | Score | Status | Reason | Answer |",
        "|---|---|---|---:|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| {pack} | {query} | {source} | {score} | {status} | {reason} | {answer} |".format(
                pack=result.case.pack,
                query=result.case.query.replace("|", "/"),
                source=result.source,
                score=result.score,
                status="PASS" if result.reason is None else "FAIL",
                reason=result.reason or "",
                answer=(result.title + ": " + result.answer).replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(report: Path) -> int:
    packs = {pack.pack_id: pack for pack in load_all_pack_contracts()}
    results: list[RetrievalResult] = []
    for case in CASES:
        match = retrieve(packs[case.pack], case.query)
        if match is None:
            results.append(RetrievalResult(case, "", "", "", 0, "no_match"))
            continue
        source, row, score = match
        answer = f"{row.title}: {row.text}"
        results.append(RetrievalResult(case, source, row.title, row.text, score, validate(case, answer)))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(results), encoding="ascii")
    passed = sum(1 for result in results if result.reason is None)
    print(f"PROBE_OK assistant_pack_retrieval_eval_cases={len(results)}")
    print(f"ASSISTANT_PACK_RETRIEVAL_EVAL|pass={passed}|total={len(results)}|report={report}")
    if passed != len(results):
        print(f"ASSISTANT_PACK_RETRIEVAL_EVAL_FAILED|pass={passed}|total={len(results)}")
        return 1
    print("PROBE_OK assistant_pack_retrieval_eval_pass=1")
    return 0


def self_test() -> None:
    assert retrieval_score("how can i ask better questions", HelpRow("ask better", "Better prompts", "Say the goal and next step.")) >= 8
    assert retrieval_score("how can i ask better questions", HelpRow("unrelated", "Other", "No match here.")) < 8
    report = markdown_report(
        [
            RetrievalResult(
                CASES[0],
                "KNOW",
                "Better prompts",
                "Say the goal, give one detail, and ask for the next useful step.",
                99,
                None,
            )
        ]
    )
    assert "Status: `PASS`" in report
    print("PROBE_OK assistant_pack_retrieval_eval_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    raise SystemExit(run_eval(args.report))


if __name__ == "__main__":
    main()

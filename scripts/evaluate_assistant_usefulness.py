#!/usr/bin/env python3
"""Evaluate assistant usefulness as operator workflows, not isolated prompts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from assistant_pack_contract import load_all_pack_contracts
from evaluate_assistant_pack_retrieval import RetrievalCase, retrieve, validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_usefulness_eval.md"


@dataclass(frozen=True)
class UsefulnessCase:
    workflow: str
    pack: str
    query: str
    operator_outcome: str
    terms: tuple[str, ...]
    min_hits: int = 3


@dataclass(frozen=True)
class UsefulnessResult:
    case: UsefulnessCase
    source: str
    score: int
    answer: str
    reason: str | None


USEFULNESS_CASES: tuple[UsefulnessCase, ...] = (
    UsefulnessCase("operator prompts", "CHAT", "how can i ask better questions", "Prompt repair gives a next actionable step.", ("goal", "detail", "next", "step")),
    UsefulnessCase("operator prompts", "CHAT", "how should i compare options", "Comparison answer names options and tradeoffs.", ("options", "tradeoff", "choose", "step")),
    UsefulnessCase("operator prompts", "CHAT", "help me plan work in small steps", "Planning answer narrows work to small verified steps.", ("small", "steps", "blocking", "verify")),
    UsefulnessCase("operator prompts", "CHAT", "what should a useful answer look like", "Answer quality is brief, concrete, and bounded.", ("brief", "concrete", "limits", "act")),
    UsefulnessCase("trust and offline limits", "CHAT", "what makes this intelligent on a small computer", "Explains intelligence as retrieval, memory, and tiny model behavior.", ("retrieval", "memory", "model", "tiny")),
    UsefulnessCase("trust and offline limits", "CHAT", "can this work without the internet", "Names local files and offline model behavior.", ("offline", "local", "files", "model")),
    UsefulnessCase("trust and offline limits", "CHAT", "what proof helps me trust this", "Points to visible local weights, tests, and logs.", ("visible", "local", "weights", "tests", "logs")),
    UsefulnessCase("trust and offline limits", "CHAT", "how do i show confidence in an answer", "Separates known, inferred, and uncertain claims.", ("known", "inferred", "uncertain"), 2),
    UsefulnessCase("dos setup and repair", "DOSHELP", "what happens before autoexec bat runs", "Explains boot order and config files.", ("config.sys", "autoexec", "drivers", "commands")),
    UsefulnessCase("dos setup and repair", "DOSHELP", "why use 8.3 filenames in batches", "Explains DOS-compatible short names.", ("8.3", "dos", "compatibility", "batch")),
    UsefulnessCase("dos setup and repair", "DOSHELP", "what should i do when cwsdpmi is missing", "Gives the protected-mode dependency fix.", ("protected-mode", "cwsdpmi.exe", "beside", "rerun")),
    UsefulnessCase("dos setup and repair", "DOSHELP", "how do i mount the dosbox bundle", "Gives a mount and profile path.", ("mount", "c:", "c:\\gpt2", "profile")),
    UsefulnessCase("dos setup and repair", "DOSHELP", "what if the fat image is full", "Explains removing training files or growing disk space.", ("training", "grow", "disk", "space"), 2),
    UsefulnessCase("dos setup and repair", "DOSHELP", "how do i handle a dos memory error", "Names conventional memory and resident-program pressure.", ("conventional", "tsrs", "drivers", "profile")),
    UsefulnessCase("hardware transfer and emulator evidence", "DOSHELP", "how should i prepare files for real hardware", "Lists the files required for physical DOS testing.", ("gpt2", "model", "packs", "cwsdpmi")),
    UsefulnessCase("hardware transfer and emulator evidence", "DOSHELP", "what logs matter from qemu", "Identifies compile/run/evidence logs as emulator proof.", ("compile", "run", "evidence", "emulator")),
    UsefulnessCase("office handoffs", "OFFICE", "how should i write a handoff note", "Produces a handoff structure with evidence and next work.", ("done", "remains", "evidence", "next")),
    UsefulnessCase("office handoffs", "OFFICE", "what belongs in a bug report", "Lists expected, actual, repro steps, and logs.", ("expected", "actual", "steps", "logs")),
    UsefulnessCase("office handoffs", "OFFICE", "make a compact release note", "Keeps release notes to change, proof, and limits.", ("changed", "proof", "limits"), 2),
    UsefulnessCase("office handoffs", "OFFICE", "what should meeting notes capture", "Tracks decisions, owners, dates, and actions.", ("decisions", "owners", "dates", "actions")),
    UsefulnessCase("office handoffs", "OFFICE", "how should i reply to a customer", "Avoids overpromising and gives status plus next step.", ("issue", "status", "next", "overpromising")),
    UsefulnessCase("office handoffs", "OFFICE", "how do i write user docs", "Frames docs around goal, prerequisites, steps, and troubleshooting.", ("goal", "prerequisites", "steps", "troubleshooting")),
    UsefulnessCase("planning and risk", "OFFICE", "help me write a project plan", "Names milestones, owners, and risks.", ("goal", "milestones", "owners", "risks")),
    UsefulnessCase("planning and risk", "OFFICE", "how do i track risks", "Captures impact, likelihood, mitigation, and owner.", ("impact", "likelihood", "mitigation", "owner")),
    UsefulnessCase("planning and risk", "OFFICE", "what is a useful test plan", "Defines scope, cases, expected result, and criteria.", ("scope", "cases", "expected", "criteria")),
    UsefulnessCase("developer pack authoring", "DEV", "how can this feel modern on a 486", "Ties modern feel to weights, retrieval, memory, and synthesis.", ("weights", "retrieval", "memory", "synthesis")),
    UsefulnessCase("developer pack authoring", "DEV", "what does retrieval first mean", "Explains KDB, USER notes, memory, and synthesis order.", ("kdb", "user", "memory", "synthesize")),
    UsefulnessCase("developer pack authoring", "DEV", "how do i author a pack", "Names HELP, KNOW, KDB, and validation.", ("help", "know", "kdb", "validator")),
    UsefulnessCase("developer pack authoring", "DEV", "what should i check before release", "Names tests, logs, checksums, and tag.", ("tests", "logs", "checksums", "tag")),
    UsefulnessCase("fast local recall architecture", "DEV", "how should we store fast recall data", "Favors compact keyworded DOS-friendly recall storage.", ("compact", "keyword", "dos", "faster")),
    UsefulnessCase("fast local recall architecture", "DEV", "what should a failure record include", "Captures command, input, expected, actual, and log.", ("command", "input", "expected", "actual", "log"), 4),
    UsefulnessCase("portable intelligence", "PORTABLE", "what does portable intelligence mean", "Frames intelligence as local weights, retrieval, memory, and offline operation.", ("local", "model", "retrieval", "memory", "network")),
    UsefulnessCase("portable intelligence", "PORTABLE", "why is basic useful for teaching ai", "Explains BASIC as an inspectable learner implementation surface.", ("basic", "arrays", "files", "integer", "inspectable")),
    UsefulnessCase("portable intelligence", "PORTABLE", "how could this move to c or assembly", "Names C, assembly, Eshkol, files, arrays, and loops as the portable contract.", ("c", "assembly", "eshkol", "files", "arrays"), 3),
    UsefulnessCase("portable intelligence", "PORTABLE", "why do hot swappable weights matter", "Explains domain behavior without rebuilding the resident runtime.", ("hot-swappable", "weights", "domain", "runtime")),
    UsefulnessCase("portable intelligence", "PORTABLE", "how should tiny machines store recall", "Keeps recall compact and indexed to reduce scanned bytes.", ("compact", "indexed", "rows", "bytes")),
    UsefulnessCase("portable intelligence", "PORTABLE", "what proof shows this works on old hardware", "Requires logs, tests, QEMU or hardware captures, and visible source.", ("logs", "tests", "qemu", "hardware", "source")),
)


def as_retrieval_case(case: UsefulnessCase) -> RetrievalCase:
    return RetrievalCase(case.pack, case.query, case.terms, case.min_hits)


def markdown_report(results: list[UsefulnessResult]) -> str:
    passed = sum(1 for result in results if result.reason is None)
    workflows = sorted({result.case.workflow for result in results})
    workflow_passes = {
        workflow: all(result.reason is None for result in results if result.case.workflow == workflow)
        for workflow in workflows
    }
    covered = sum(1 for passed_workflow in workflow_passes.values() if passed_workflow)
    status = "PASS" if passed == len(results) and covered == len(workflows) else "FAIL"
    lines = [
        "# Assistant Usefulness Evaluation",
        "",
        f"Status: `{status}`",
        f"Task pass rate: `{passed}/{len(results)}`",
        f"Workflow coverage: `{covered}/{len(workflows)}`",
        "",
        "This gate groups assistant behavior into operator workflows. It checks whether local packs can produce actionable answers from local recall without network access.",
        "",
        "| Workflow | Pack | Operator Task | Status | Reason | Outcome | Source | Score | Answer |",
        "|---|---|---|---|---|---|---|---:|---|",
    ]
    for result in results:
        lines.append(
            "| {workflow} | {pack} | {query} | {status} | {reason} | {outcome} | {source} | {score} | {answer} |".format(
                workflow=result.case.workflow.replace("|", "/"),
                pack=result.case.pack,
                query=result.case.query.replace("|", "/"),
                status="PASS" if result.reason is None else "FAIL",
                reason=result.reason or "",
                outcome=result.case.operator_outcome.replace("|", "/"),
                source=result.source,
                score=result.score,
                answer=result.answer.replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(report: Path) -> int:
    packs = {pack.pack_id: pack for pack in load_all_pack_contracts()}
    results: list[UsefulnessResult] = []
    for case in USEFULNESS_CASES:
        match = retrieve(packs[case.pack], case.query)
        if match is None:
            results.append(UsefulnessResult(case, "", 0, "", "no_match"))
            continue
        source, row, score = match
        answer = f"{row.title}: {row.text}"
        results.append(UsefulnessResult(case, source, score, answer, validate(as_retrieval_case(case), answer)))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(results), encoding="ascii")
    passed = sum(1 for result in results if result.reason is None)
    workflows = {result.case.workflow for result in results}
    covered = sum(
        1
        for workflow in workflows
        if all(result.reason is None for result in results if result.case.workflow == workflow)
    )
    print(f"PROBE_OK assistant_usefulness_eval_cases={len(results)}")
    print(f"ASSISTANT_USEFULNESS_EVAL|pass={passed}|total={len(results)}|workflows={covered}/{len(workflows)}|report={report}")
    if passed != len(results) or covered != len(workflows):
        print(f"ASSISTANT_USEFULNESS_EVAL_FAILED|pass={passed}|total={len(results)}|workflows={covered}/{len(workflows)}")
        return 1
    print("PROBE_OK assistant_usefulness_eval_pass=1")
    return 0


def self_test() -> None:
    assert len(USEFULNESS_CASES) >= 30
    assert len({case.workflow for case in USEFULNESS_CASES}) >= 8
    report = markdown_report(
        [
            UsefulnessResult(
                USEFULNESS_CASES[0],
                "KDB",
                99,
                "Better prompts: Say the goal, give one detail, and ask for the next useful step.",
                None,
            )
        ]
    )
    assert "Status: `PASS`" in report
    assert "Workflow coverage: `1/1`" in report
    print("PROBE_OK assistant_usefulness_eval_self_test=1")


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

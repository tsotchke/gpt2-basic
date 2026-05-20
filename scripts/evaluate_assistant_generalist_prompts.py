#!/usr/bin/env python3
"""Evaluate the CHAT model on broad generalist prompt phrasings.

This gate is separate from the pack quality report because it intentionally
tracks non-demo conversational categories that previously failed, including
troubleshooting, trust, offline limits, memory, and release checks.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from evaluate_assistant_raw_prompts import RawPromptCase, clean_completion, validate_completion
from evaluate_gpt2_basic_quality import QualityPrompt, evaluate_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "gpt2_basic" / "PACKS" / "CHAT" / "MODEL"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_generalist_prompt_eval.md"


@dataclass(frozen=True)
class GeneralistCase:
    query: str
    terms: tuple[str, ...]
    min_hits: int = 2


GENERALIST_CASES: tuple[GeneralistCase, ...] = (
    GeneralistCase("guide me through troubleshooting a failure", ("error", "change", "test", "step")),
    GeneralistCase("what first step should i take when debugging fails", ("error", "test", "step", "fix")),
    GeneralistCase("i am overwhelmed and need one next task", ("small", "task", "step")),
    GeneralistCase("my todo list is too big", ("small", "task", "step"), 1),
    GeneralistCase("why should someone believe this is real", ("local", "model", "weights", "dos")),
    GeneralistCase("how can i tell this is not fake", ("real", "local", "model", "weights")),
    GeneralistCase("what if my question is too long", ("short", "prompt", "question", "dos")),
    GeneralistCase("should i type a paragraph or a short prompt", ("short", "prompt", "dos")),
    GeneralistCase("can you talk about history a little", ("question", "topic", "simple")),
    GeneralistCase("help me learn a new topic", ("question", "topic", "simple"), 1),
    GeneralistCase("write a kind line about vintage dos hardware", ("old", "dos", "computer", "local", "model")),
    GeneralistCase("say something friendly about old dos hardware", ("old", "dos", "computer", "local", "model")),
    GeneralistCase("what does local mean in this project", ("local", "machine", "model")),
    GeneralistCase("explain local in this demo", ("local", "machine", "model")),
    GeneralistCase("do you keep memory during the session", ("remember", "session", "facts")),
    GeneralistCase("can you remember small facts here", ("remember", "session", "facts")),
    GeneralistCase("how can i stop looped output", ("short", "prompt", "reset", "loop")),
    GeneralistCase("what should i do if output loops", ("short", "prompt", "reset", "loop")),
    GeneralistCase("give me a short release checklist", ("tag", "assets", "checksums", "tests"), 3),
    GeneralistCase("what is a small release checklist", ("tag", "assets", "checksums", "tests"), 3),
    GeneralistCase("explain emulator simply", ("emulator", "machine", "another")),
    GeneralistCase("what does an emulator do", ("emulator", "machine", "another")),
    GeneralistCase("why can this not go online", ("dos", "browse", "local", "files")),
    GeneralistCase("why is this dos chat offline", ("dos", "browse", "local", "files")),
)


def case_name(case: GeneralistCase) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", case.query.lower()).strip("_")
    return f"chat_generalist_{normalized[:48]}"


def prompt_text(case: GeneralistCase) -> str:
    return f"User: {case.query} Assistant:"


def as_raw_case(case: GeneralistCase) -> RawPromptCase:
    return RawPromptCase("CHAT", case.query, case.terms, case.min_hits)


def markdown_report(rows: list[tuple[GeneralistCase, str, str | None]]) -> str:
    pass_count = sum(1 for _case, _completion, reason in rows if reason is None)
    status = "PASS" if pass_count == len(rows) else "NEEDS_TRAINING"
    lines = [
        "# Assistant Generalist Prompt Evaluation",
        "",
        f"Status: `{status}`",
        f"Prompt pass rate: `{pass_count}/{len(rows)}`",
        "",
        "These prompts exercise CHAT categories such as troubleshooting, trust, memory, offline limits, release checks, and old-hardware context.",
        "",
        "| Query | Status | Reason | Completion |",
        "|---|---|---|---|",
    ]
    for case, completion, reason in rows:
        lines.append(
            "| {query} | {status} | {reason} | {completion} |".format(
                query=case.query.replace("|", "/"),
                status="PASS" if reason is None else "FAIL",
                reason=reason or "",
                completion=clean_completion(completion).replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(args: argparse.Namespace) -> int:
    cases = list(GENERALIST_CASES)
    if args.limit > 0:
        cases = cases[: args.limit]
    prompts = [QualityPrompt(case_name(case), prompt_text(case), case.terms) for case in cases]
    _cfg, results = evaluate_model(
        args.model_dir,
        prompts,
        args.max_new_tokens,
        args.min_generated,
        args.threshold,
        args.backend,
        args.device,
        "assistant_generalist_chat",
    )
    rows = [
        (case, result.completion, validate_completion(as_raw_case(case), result.completion))
        for case, result in zip(cases, results)
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(rows), encoding="ascii")
    pass_count = sum(1 for _case, _completion, reason in rows if reason is None)
    print(f"PROBE_OK assistant_generalist_prompt_eval_cases={len(rows)}")
    print(f"ASSISTANT_GENERALIST_PROMPT_EVAL|pass={pass_count}|total={len(rows)}|report={args.report}")
    if pass_count != len(rows):
        print(f"ASSISTANT_GENERALIST_PROMPT_EVAL_FAILED|pass={pass_count}|total={len(rows)}")
        return 1
    print("PROBE_OK assistant_generalist_prompt_eval_pass=1")
    return 0


def self_test() -> None:
    assert len(GENERALIST_CASES) >= 24
    assert "troubleshooting" in GENERALIST_CASES[0].query
    good = "Check the first error, change one thing, then test again."
    bad = "Use two brief sentences."
    first = as_raw_case(GENERALIST_CASES[0])
    assert validate_completion(first, good) is None
    assert validate_completion(first, bad) is not None
    report = markdown_report([(GENERALIST_CASES[0], good, None)])
    assert "Status: `PASS`" in report
    print("PROBE_OK assistant_generalist_prompt_eval_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--backend", choices=("float", "fixed"), default="float")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-generated", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    status = run_eval(args)
    if status != 0 and not args.allow_fail:
        raise SystemExit(status)


if __name__ == "__main__":
    main()

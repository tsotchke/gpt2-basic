#!/usr/bin/env python3
"""Evaluate pack assistant models on original held-out chat prompts.

This gate measures the raw pack-local model, before ASSIST.EXE retrieval or
fallback repair. It is intentionally stricter than the release stress probe:
the goal is to catch prompt leakage, repetition, topic drift, and brittle
answers in the model itself.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from evaluate_gpt2_basic_quality import QualityPrompt, boundary_error_count, evaluate_model
from stress_assistant_behavior import bad_visible_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK_ROOT = ROOT / "assets" / "gpt2_basic" / "PACKS"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_raw_prompt_eval.md"


@dataclass(frozen=True)
class RawPromptCase:
    pack: str
    query: str
    terms: tuple[str, ...]


RAW_PROMPT_CASES: tuple[RawPromptCase, ...] = (
    RawPromptCase("CHAT", "why are you saying the same phrase again", ("repeat", "short", "retry", "reset")),
    RawPromptCase("CHAT", "explain local inference without jargon", ("local", "inference", "model", "machine")),
    RawPromptCase("CHAT", "is this answer coming from real model weights", ("real", "model", "weights", "local")),
    RawPromptCase("CHAT", "what should i do if the output looks wrong", ("short", "retry", "question", "switch")),
    RawPromptCase("CHAT", "give me a three step bug fixing plan", ("bug", "step", "error", "test")),
    RawPromptCase("CHAT", "what is the difference between my prompt and your answer", ("prompt", "answer", "question", "output")),
    RawPromptCase("CHAT", "why does an old computer demo matter", ("old", "computer", "dos", "hardware", "local")),
    RawPromptCase("CHAT", "tell me whether the release is ready", ("release", "tag", "asset", "test", "checksum")),
    RawPromptCase("CHAT", "i am stuck and need a small next step", ("step", "small", "test", "adjust")),
    RawPromptCase("CHAT", "can you make a tiny plan for today", ("plan", "goal", "step", "task")),
    RawPromptCase("CHAT", "tell me a short story about a dos model", ("story", "dos", "model", "prompt")),
    RawPromptCase("CHAT", "what are your limits in this session", ("tiny", "local", "limit", "session")),
    RawPromptCase("CHAT", "can you browse the internet from dos", ("internet", "browse", "dos", "cannot")),
    RawPromptCase("CHAT", "what does a token mean", ("token", "text", "piece")),
    RawPromptCase("CHAT", "how do i focus for a few minutes", ("focus", "task", "distraction", "step")),
    RawPromptCase("CHAT", "write one friendly sentence about this demo", ("demo", "dos", "friendly", "local")),
    RawPromptCase("DOSHELP", "why does my protected mode program need cwsdpmi", ("dpmi", "protected", "cwsdpmi")),
    RawPromptCase("DOSHELP", "how do i leave more conventional memory free", ("memory", "himem", "umb", "conventional")),
    RawPromptCase("DOSHELP", "what should i put in config.sys for this assistant", ("config.sys", "himem", "files", "buffers")),
    RawPromptCase("DOSHELP", "write a safe batch check for the model directory", ("if exist", "batch", "model", "8.3")),
    RawPromptCase("DOSHELP", "my autoexec is messy and slow", ("autoexec", "path", "resident", "short")),
    RawPromptCase("OFFICE", "rewrite this politely: the artifact failed", ("polite", "direct", "artifact")),
    RawPromptCase("OFFICE", "summarize this: tests passed but the tag was stale", ("summary", "tests", "tag")),
    RawPromptCase("OFFICE", "make this clearer: checksums changed after rebuild", ("happened", "matters", "checksum")),
    RawPromptCase("OFFICE", "shorten this sentence without losing the decision", ("short", "decision", "remove", "duplicate")),
    RawPromptCase("OFFICE", "write a status update about a delayed build", ("status", "polite", "concrete", "action")),
)


def clean_completion(text: str) -> str:
    lower = text.lower()
    cut_pos = len(text)
    for marker in (" user:", " assistant:", " note:", " prompt:", " reply:", " q:", " a:"):
        marker_pos = lower.find(marker)
        if marker_pos >= 0:
            cut_pos = min(cut_pos, marker_pos)
    return text[:cut_pos].strip()


def prompt_text(case: RawPromptCase) -> str:
    return f"User: {case.query} Assistant:"


def case_name(case: RawPromptCase) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", case.query.lower()).strip("_")
    return f"{case.pack.lower()}_{normalized[:44]}"


def relevant(case: RawPromptCase, text: str) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in case.terms)


def validate_completion(case: RawPromptCase, completion: str) -> str | None:
    cleaned = clean_completion(completion)
    bad = bad_visible_text(cleaned)
    if bad is not None:
        return bad
    if boundary_error_count(cleaned) > 0:
        return "boundary_error"
    if cleaned.lower() == case.query.lower():
        return "query_echo"
    if not relevant(case, cleaned):
        return "irrelevant"
    return None


def markdown_report(rows: list[tuple[RawPromptCase, str, str | None]]) -> str:
    pass_count = sum(1 for _case, _completion, reason in rows if reason is None)
    status = "PASS" if pass_count == len(rows) else "NEEDS_TRAINING"
    lines = [
        "# Assistant Raw Prompt Evaluation",
        "",
        f"Status: `{status}`",
        f"Prompt pass rate: `{pass_count}/{len(rows)}`",
        "",
        "| Pack | Query | Status | Reason | Completion |",
        "|---|---|---|---|---|",
    ]
    for case, completion, reason in rows:
        cleaned = clean_completion(completion).replace("|", "/")
        lines.append(
            "| {pack} | {query} | {status} | {reason} | {completion} |".format(
                pack=case.pack,
                query=case.query.replace("|", "/"),
                status="PASS" if reason is None else "FAIL",
                reason=reason or "",
                completion=cleaned,
            )
        )
    lines.append("")
    return "\n".join(lines)


def selected_cases(pack_filter: list[str] | None, limit: int) -> list[RawPromptCase]:
    cases = list(RAW_PROMPT_CASES)
    if pack_filter:
        wanted = {pack.upper() for pack in pack_filter}
        cases = [case for case in cases if case.pack in wanted]
    if limit > 0:
        cases = cases[:limit]
    return cases


def run_eval(args: argparse.Namespace) -> int:
    cases = selected_cases(args.pack, args.limit)
    if not cases:
        raise SystemExit("no raw prompt cases selected")

    rows: list[tuple[RawPromptCase, str, str | None]] = []
    for pack_id in sorted({case.pack for case in cases}):
        pack_cases = [case for case in cases if case.pack == pack_id]
        model_dir = args.pack_root / pack_id / "MODEL"
        prompts = [
            QualityPrompt(case_name(case), prompt_text(case), case.terms)
            for case in pack_cases
        ]
        _cfg, results = evaluate_model(
            model_dir,
            prompts,
            args.max_new_tokens,
            args.min_generated,
            args.threshold,
            args.backend,
            args.device,
            f"assistant_raw_{pack_id.lower()}",
        )
        for case, result in zip(pack_cases, results):
            rows.append((case, result.completion, validate_completion(case, result.completion)))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(rows), encoding="ascii")
    pass_count = sum(1 for _case, _completion, reason in rows if reason is None)
    print(f"PROBE_OK assistant_raw_prompt_eval_cases={len(rows)}")
    print(f"ASSISTANT_RAW_PROMPT_EVAL|pass={pass_count}|total={len(rows)}|report={args.report}")
    if pass_count != len(rows):
        print(f"ASSISTANT_RAW_PROMPT_EVAL_FAILED|pass={pass_count}|total={len(rows)}")
        return 1
    print("PROBE_OK assistant_raw_prompt_eval_pass=1")
    return 0


def self_test() -> None:
    good = "Local inference means the model runs on this machine."
    leak = "Use two brief sentences."
    repeat = "tag tag tag tag tag."
    case = RawPromptCase("CHAT", "explain local inference", ("local", "inference", "model"))
    assert validate_completion(case, good) is None
    assert validate_completion(case, leak) is not None
    assert validate_completion(case, repeat) is not None
    assert "Assistant:" in prompt_text(case)
    assert len(selected_cases(["CHAT"], 0)) >= 16
    print("PROBE_OK assistant_raw_prompt_eval_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--backend", choices=("float", "fixed"), default="float")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-generated", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--pack", action="append")
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

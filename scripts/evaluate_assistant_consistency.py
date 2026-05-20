#!/usr/bin/env python3
"""Run a stricter consistency evaluation for assistant pack model output.

This extends the raw prompt gate by testing each held-out prompt through
multiple phrasing variants. It still evaluates the pack-local language models
directly, before ASSIST.EXE retrieval, golden replies, memory, or fallback
repair can hide weak model output.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from evaluate_assistant_raw_prompts import (
    DEFAULT_PACK_ROOT,
    RAW_PROMPT_CASES,
    RawPromptCase,
    clean_completion,
    validate_completion,
)
from evaluate_gpt2_basic_quality import QualityPrompt, evaluate_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_consistency_eval.md"


@dataclass(frozen=True)
class ConsistencyCase:
    base: RawPromptCase
    variant_name: str
    query: str
    model_query: str


def sentence_base(text: str) -> str:
    return text.strip().rstrip(".?!").strip()


def query_variants(query: str) -> tuple[tuple[str, str], ...]:
    base = sentence_base(query)
    return (
        ("base", base),
        ("question_mark", base + "?"),
        ("please", "please answer this: " + base),
        ("short", "short answer please: " + base),
        ("typed", "i typed this question: " + base),
        ("dos_chat", "dos chat question: " + base),
    )


def canonical_query(query: str) -> str:
    cleaned = query.strip()
    lower = cleaned.lower()
    prefixes = (
        "please answer this:",
        "short answer please:",
        "i typed this question:",
        "dos chat question:",
        "help me with this:",
    )
    for prefix in prefixes:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return sentence_base(cleaned)


def case_name(case: ConsistencyCase) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", case.base.query.lower()).strip("_")
    return f"{case.base.pack.lower()}_{normalized[:38]}_{case.variant_name}"


def prompt_text(case: ConsistencyCase) -> str:
    return f"User: {case.model_query} Assistant:"


def selected_base_cases(pack_filter: list[str] | None, limit: int) -> list[RawPromptCase]:
    cases = list(RAW_PROMPT_CASES)
    if pack_filter:
        wanted = {pack.upper() for pack in pack_filter}
        cases = [case for case in cases if case.pack in wanted]
    if limit > 0:
        cases = cases[:limit]
    return cases


def expand_cases(base_cases: list[RawPromptCase], variants_per_prompt: int, canonicalize: bool) -> list[ConsistencyCase]:
    expanded: list[ConsistencyCase] = []
    for base_case in base_cases:
        variants = query_variants(base_case.query)
        if variants_per_prompt > 0:
            variants = variants[:variants_per_prompt]
        for variant_name, query in variants:
            model_query = canonical_query(query) if canonicalize else query
            expanded.append(ConsistencyCase(base_case, variant_name, query, model_query))
    return expanded


def markdown_report(rows: list[tuple[ConsistencyCase, str, str | None]], backend: str, canonicalize: bool) -> str:
    pass_count = sum(1 for _case, _completion, reason in rows if reason is None)
    groups: dict[tuple[str, str], list[tuple[ConsistencyCase, str, str | None]]] = defaultdict(list)
    pack_counts: Counter[str] = Counter()
    pack_passes: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for case, completion, reason in rows:
        groups[(case.base.pack, case.base.query)].append((case, completion, reason))
        pack_counts[case.base.pack] += 1
        if reason is None:
            pack_passes[case.base.pack] += 1
        else:
            reason_counts[reason] += 1

    consistent_groups = sum(1 for group in groups.values() if all(reason is None for _case, _completion, reason in group))
    status = "PASS" if pass_count == len(rows) and consistent_groups == len(groups) else "NEEDS_TRAINING"

    lines = [
        "# Assistant Consistency Evaluation",
        "",
        f"Status: `{status}`",
        f"Backend: `{backend}`",
        f"Prompt canonicalization: `{'on' if canonicalize else 'off'}`",
        f"Prompt variants: `{pass_count}/{len(rows)}`",
        f"Consistent prompt groups: `{consistent_groups}/{len(groups)}`",
        "",
        "## Pack Summary",
        "",
        "| Pack | Prompt Variants | Pass Rate |",
        "|---|---:|---:|",
    ]
    for pack in sorted(pack_counts):
        lines.append(f"| {pack} | {pack_counts[pack]} | {pack_passes[pack]}/{pack_counts[pack]} |")

    lines.extend(["", "## Failure Reasons", ""])
    if reason_counts:
        lines.extend(f"- `{reason}`: `{count}`" for reason, count in sorted(reason_counts.items()))
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Prompt Variants",
            "",
            "| Pack | Base Query | Variant | Model Query | Status | Reason | Completion |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for case, completion, reason in rows:
        lines.append(
            "| {pack} | {base} | {variant} | {model_query} | {status} | {reason} | {completion} |".format(
                pack=case.base.pack,
                base=case.base.query.replace("|", "/"),
                variant=case.variant_name,
                model_query=case.model_query.replace("|", "/"),
                status="PASS" if reason is None else "FAIL",
                reason=reason or "",
                completion=clean_completion(completion).replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(args: argparse.Namespace) -> int:
    base_cases = selected_base_cases(args.pack, args.limit)
    if not base_cases:
        raise SystemExit("no consistency cases selected")
    cases = expand_cases(base_cases, args.variants, not args.no_canonicalize)
    rows: list[tuple[ConsistencyCase, str, str | None]] = []

    for pack_id in sorted({case.base.pack for case in cases}):
        pack_cases = [case for case in cases if case.base.pack == pack_id]
        prompts = [
            QualityPrompt(case_name(case), prompt_text(case), case.base.terms)
            for case in pack_cases
        ]
        _cfg, results = evaluate_model(
            args.pack_root / pack_id / "MODEL",
            prompts,
            args.max_new_tokens,
            args.min_generated,
            args.threshold,
            args.backend,
            args.device,
            f"assistant_consistency_{pack_id.lower()}",
        )
        for case, result in zip(pack_cases, results):
            rows.append((case, result.completion, validate_completion(case.base, result.completion)))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(rows, args.backend, not args.no_canonicalize), encoding="ascii")
    pass_count = sum(1 for _case, _completion, reason in rows if reason is None)
    groups: dict[tuple[str, str], list[str | None]] = defaultdict(list)
    for case, _completion, reason in rows:
        groups[(case.base.pack, case.base.query)].append(reason)
    consistent_groups = sum(1 for reasons in groups.values() if all(reason is None for reason in reasons))

    print(f"PROBE_OK assistant_consistency_cases={len(rows)}")
    print(f"ASSISTANT_CONSISTENCY_EVAL|pass={pass_count}|total={len(rows)}|groups={consistent_groups}/{len(groups)}|report={args.report}")
    if pass_count != len(rows) or consistent_groups != len(groups):
        print(f"ASSISTANT_CONSISTENCY_EVAL_FAILED|pass={pass_count}|total={len(rows)}|groups={consistent_groups}/{len(groups)}")
        return 1
    print("PROBE_OK assistant_consistency_eval_pass=1")
    return 0


def self_test() -> None:
    first = RAW_PROMPT_CASES[0]
    expanded = expand_cases([first], 6, True)
    assert len(expanded) == 6
    assert expanded[0].variant_name == "base"
    assert expanded[1].variant_name == "question_mark"
    assert expanded[1].model_query == first.query
    assert expanded[-1].query.startswith("dos chat question:")
    assert expanded[-1].model_query == first.query
    assert prompt_text(expanded[0]).startswith("User: ")
    assert case_name(expanded[0]).endswith("_base")
    text = markdown_report([(expanded[0], "Hello from DOS.", None)], "float", True)
    assert "Status: `PASS`" in text
    assert "Consistent prompt groups: `1/1`" in text
    print("PROBE_OK assistant_consistency_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--backend", choices=("float", "fixed"), default="float")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-generated", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--pack", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-canonicalize", action="store_true")
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

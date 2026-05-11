#!/usr/bin/env python3
"""Score the GPT2-BASIC DOS fixed-point quality-suite log."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_gpt2_basic_quality import (
    PROMPT_SUITES,
    QualityResult,
    quality_prompts,
    score_completion,
)
from export_gpt2_basic_vectors import parse_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
DEFAULT_LOG = ROOT / "qemu" / "evidence" / "quality_486.log"
DEFAULT_OUTPUT = ROOT / "qemu" / "evidence" / "quality_report_dos.md"


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def default_output_for_suite(suite: str) -> Path:
    if suite == "heldout":
        return ROOT / "qemu" / "evidence" / "quality_report_dos_heldout.md"
    if suite == "all":
        return ROOT / "qemu" / "evidence" / "quality_report_dos_all.md"
    return DEFAULT_OUTPUT


def clean_line(line: str) -> str:
    return line.replace("\x08", "").strip()


def parse_generated_blocks(log_path: Path) -> dict[str, str]:
    blocks: dict[str, str] = {}
    current_name = ""
    prompt = ""
    in_generated = False
    waiting_for_separator = False
    captured: list[str] = []

    for raw_line in log_path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = clean_line(raw_line)
        if line.startswith("QUALITY_PROMPT_BEGIN|"):
            if current_name and captured:
                blocks[current_name] = "\n".join(captured).strip()
            fields = line.split("|", 2)
            current_name = fields[1] if len(fields) > 1 else ""
            prompt = fields[2] if len(fields) > 2 else ""
            in_generated = False
            waiting_for_separator = False
            captured = []
            continue

        if line.startswith("QUALITY_PROMPT_END|"):
            if current_name:
                text = "\n".join(captured).strip()
                if text.startswith(prompt):
                    text = text[len(prompt) :]
                blocks[current_name] = text.strip()
            current_name = ""
            prompt = ""
            in_generated = False
            waiting_for_separator = False
            captured = []
            continue

        if not current_name:
            continue

        if line == "Generated Text:":
            waiting_for_separator = True
            in_generated = False
            continue

        if waiting_for_separator and set(line) == {"-"}:
            waiting_for_separator = False
            in_generated = True
            continue

        if in_generated and set(line) == {"-"}:
            in_generated = False
            continue

        if in_generated:
            captured.append(line)

    return blocks


def markdown_report(cfg, results: list[QualityResult], threshold: float, log_path: Path, suite: str) -> str:
    avg_score = sum(result.score for result in results) / max(1, len(results))
    pass_count = sum(1 for result in results if result.passed)
    status = "PASS" if pass_count == len(results) else "NEEDS_TRAINING"

    lines = [
        "# GPT2-BASIC DOS Fixed-Point Quality Report",
        "",
        f"Model profile: `{cfg.profile}`",
        f"Shape: `{cfg.n_layer}L {cfg.n_embd}D {cfg.n_head}H ctx{cfg.n_positions} hidden{cfg.hidden_dim} vocab{cfg.vocab_size}`",
        "Evaluation backend: `dos-fixed-qemu`",
        f"Quality suite: `{suite}`",
        f"Source log: `{display_path(log_path)}`",
        f"Quality status: `{status}`",
        f"Average score: `{avg_score:.3f}`",
        f"Prompt pass rate: `{pass_count}/{len(results)}` at threshold `{threshold:.2f}`",
        "",
        "## Prompt Suite",
        "",
        "| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    f"{result.score:.3f}",
                    str(result.keyword_hits),
                    f"{result.repeated_trigram_ratio * 100.0:.1f}%",
                    str(result.max_char_run),
                    str(result.boundary_errors),
                    "yes" if result.ended_cleanly else "no",
                    "PASS" if result.passed else "RETRAIN",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Generated Continuations", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.name}",
                "",
                f"Prompt: `{result.prompt}`",
                "",
                "```text",
                result.completion.replace("|", "/"),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def self_test(model_dir: Path, log_path: Path, threshold: float) -> None:
    cfg = parse_config(model_dir / "GPT2CFG.TXT")
    blocks = parse_generated_blocks(log_path)
    prompts = quality_prompts("runtime-regression")
    results: list[QualityResult] = []
    for prompt in prompts:
        if prompt.name not in blocks:
            raise ValueError(f"{log_path} does not contain quality block {prompt.name!r}")
        results.append(score_completion(prompt, blocks[prompt.name], len(blocks[prompt.name]), threshold))
    report = markdown_report(cfg, results, threshold, log_path, "runtime-regression")
    print("trace_scope dos_quality_log_contract")
    print("trace parse_generated_blocks")
    print("trace markdown_report")
    print(f"PROBE_OK parse_generated_blocks blocks={len(blocks)}")
    print(f"PROBE_OK dos_quality_report bytes={len(report)}")
    print("PROBE_OK evaluate_dos_quality_log self_test=1")
    print("PROBE_OK main cli_entry=available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--suite", choices=sorted(PROMPT_SUITES), default="runtime-regression")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.model_dir, args.log, args.threshold)
        return

    cfg = parse_config(args.model_dir / "GPT2CFG.TXT")
    blocks = parse_generated_blocks(args.log)
    results: list[QualityResult] = []
    for prompt in quality_prompts(args.suite):
        if prompt.name not in blocks:
            raise ValueError(f"{args.log} does not contain quality block {prompt.name!r}")
        completion = blocks[prompt.name]
        results.append(score_completion(prompt, completion, len(completion), args.threshold))

    report = markdown_report(cfg, results, args.threshold, args.log, args.suite)
    output = args.output if args.output != DEFAULT_OUTPUT or args.suite == "runtime-regression" else default_output_for_suite(args.suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="ascii")
    print(report)


if __name__ == "__main__":
    main()

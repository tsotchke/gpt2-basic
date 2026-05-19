#!/usr/bin/env python3
"""Write a concrete repair plan for exported GPT2-BASIC model quality."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from audit_exported_models import (
    DEFAULT_EVIDENCE,
    DEFAULT_MODELS_ROOT,
    DEFAULT_PACK_ROOT,
    AuditRow,
    audit_models,
    required_quality,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "model_quality_repair_plan.md"


@dataclass(frozen=True)
class RepairAction:
    model: str
    decision: str
    reason: str
    command: str


RELEASE_MODEL_NAMES = {
    "MODEL",
    "MODEL_LEXICON_GOLD_V4_S3000",
    "MODEL_HEADQ4_PROD_PROBE",
    "MODEL_HEADSHORTLIST2048_PROD_PROBE",
    "MODEL_TOKHEADQ4_PROD_PROBE",
    "MODEL_TOKHEADQ4_STREAM_PROD_PROBE",
}


def quality_text(row: AuditRow) -> str:
    report = required_quality(row.quality_reports)
    if report is None:
        return "missing"
    return f"{report.status} {report.passed}/{report.total} avg {report.average:.3f} ({report.backend}, {report.suite})"


def train_command(output: str, profile: str, tokenizer: str, vocab_size: int, steps: int, corpus_weight: int = 4) -> str:
    return (
        "python3 scripts/train_tiny_gpt.py "
        f"--profile {profile} --tokenizer {tokenizer} --vocab-size {vocab_size} "
        "--include-docs --corpus-file data/domain_curriculum/gold_curriculum_v4.txt "
        f"--corpus-weight {corpus_weight} --steps {steps} --log-every 500 "
        f"--output assets/gpt2_basic/{output}"
    )


def clean_486dx2_repair_command(output: str) -> str:
    return (
        "python3 scripts/train_tiny_gpt.py "
        "--profile 486dx2-usable --tokenizer lexicon --vocab-size 4096 "
        "--lexicon-min-count 1 --base-weight 0 "
        "--corpus-file data/domain_curriculum/gold_curriculum_v5_clean_repair.txt "
        "--corpus-weight 1 --steps 7000 --log-every 500 "
        f"--output assets/gpt2_basic/{output}"
    )


def action_for(row: AuditRow) -> RepairAction:
    name = row.model.name
    report = required_quality(row.quality_reports)
    passing = bool(report and report.status == "PASS" and report.passed == report.total)

    if row.model.role == "assistant_pack":
        return RepairAction(name, "keep", "assistant pack model passes its pack-local strict gate", "")
    if passing and name in RELEASE_MODEL_NAMES:
        return RepairAction(name, "keep_release", "release-shaped model passes the strict all-suite gate", "")
    if passing and name == "MODEL_LEXICON_GOLD_V2_S3000":
        return RepairAction(
            name,
            "retire_superseded",
            "gold-v2 is quality-proven but superseded by gold-v4 default evidence",
            "",
        )
    if passing:
        return RepairAction(name, "keep_candidate", "candidate passes host strict all-suite; needs DOS vector/perf evidence before promotion", "")
    if "REPAIR" in name:
        return RepairAction(name, "failed_repair", "repair attempt has artifact evidence but still fails strict all-suite quality", "")
    if name == "MODEL_SUBWORD512_PROTO":
        return RepairAction(name, "reject_host_only", "host-only tokenizer prototype is not DOS-ready and fails quality", "")
    if name == "MODEL_PROFILE_386_MIN":
        return RepairAction(
            name,
            "retrain_priority",
            "fastest measured profile is valuable only if a small-vocabulary repair passes quality",
            train_command("MODEL_PROFILE_386_MIN_LEXICON384_REPAIR", "386-min", "lexicon", 384, 5000),
        )
    if name == "MODEL_PROFILE_486DX2_USABLE":
        return RepairAction(
            name,
            "retrain_priority",
            "larger 486DX2 profile should be retrained from the clean gold curriculum, not old byte data",
            clean_486dx2_repair_command("MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR2"),
        )
    if name.startswith("MODEL_BPE"):
        return RepairAction(
            name,
            "retrain_experiment",
            "BPE path is an experiment; retrain only if comparing tokenizer families is still useful",
            train_command("MODEL_BPE384_GOLD_V4_REPAIR", "486sx-safe", "bpe", 384, 5000),
        )
    if "LEXICON4096" in name and "ADAPTED" in name:
        return RepairAction(name, "retire_superseded", "adapted/repair branch is superseded by the gold-v4 release path", "")
    if "POSTTRAIN" in name:
        return RepairAction(name, "retire_superseded", "post-training repair damaged fluency; keep as negative evidence only", "")
    if "BASELINE" in name or "CANDIDATE" in name or "DOMAIN" in name or "ONLINE" in name:
        return RepairAction(name, "retire_superseded", "old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models", "")
    return RepairAction(name, "retrain_optional", "fails strict all-suite but has no release-critical role", "")


def markdown(rows: list[AuditRow]) -> str:
    actions = [action_for(row) for row in rows]
    counts: dict[str, int] = {}
    for action in actions:
        counts[action.decision] = counts.get(action.decision, 0) + 1

    lines = [
        "# Model Quality Repair Plan",
        "",
        "The strict gate requires all-suite quality when available. Historical candidates that fail this gate should not be treated as latent release options.",
        "",
        "## Summary",
        "",
    ]
    for decision in sorted(counts):
        lines.append(f"- `{decision}`: {counts[decision]}")

    lines.extend(
        [
            "",
            "## Actions",
            "",
            "| Model | Quality | Decision | Reason |",
            "|---|---|---|---|",
        ]
    )
    for row, action in zip(rows, actions):
        lines.append(f"| `{action.model}` | {quality_text(row)} | `{action.decision}` | {action.reason} |")

    commands: list[RepairAction] = []
    seen_commands: set[str] = set()
    for action in actions:
        if not action.command or action.command in seen_commands:
            continue
        commands.append(action)
        seen_commands.add(action.command)
    lines.extend(["", "## Retrain Commands", ""])
    if not commands:
        lines.append("No retrain commands generated.")
    else:
        for action in commands:
            lines.extend(["```sh", action.command, "```", ""])
            output_name = action.command.rsplit(" ", 1)[-1].removeprefix("assets/gpt2_basic/")
            lines.extend(
                [
                    "```sh",
                    (
                        "python3 scripts/evaluate_gpt2_basic_quality.py "
                        f"--model-dir assets/gpt2_basic/{output_name} "
                        "--suite all --backend float --threshold 0.72 "
                        f"--output qemu/evidence/quality_report_{output_name.lower().removeprefix('model_')}_all.md"
                    ),
                    "```",
                    "",
                ]
            )

    lines.append("")
    return "\n".join(lines)


def self_test(rows: list[AuditRow]) -> None:
    text = markdown(rows[:3])
    if "Model Quality Repair Plan" not in text:
        raise RuntimeError("repair plan markdown missing title")
    print(f"PROBE_OK model_quality_repair_plan_rows={len(rows)}")
    print("PROBE_OK model_quality_repair_plan_markdown=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    rows = audit_models(args.models_root, args.pack_root, args.evidence_dir, refresh_model_reports=False)
    if args.self_test:
        self_test(rows)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(rows), encoding="ascii")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

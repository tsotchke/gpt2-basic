#!/usr/bin/env python3
"""Write the current GPT2-BASIC improvement backlog from evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from audit_exported_models import (
    DEFAULT_EVIDENCE,
    DEFAULT_MODELS_ROOT,
    DEFAULT_PACK_ROOT,
    AuditRow,
    audit_models,
    required_quality,
)
from plan_model_quality_repairs import action_for, quality_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "improvement_backlog.md"


def count_decisions(rows: list[AuditRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        decision = action_for(row).decision
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def pass_count(rows: list[AuditRow]) -> int:
    total = 0
    for row in rows:
        report = required_quality(row.quality_reports)
        if report is not None and report.status == "PASS" and report.passed == report.total:
            total += 1
    return total


def model_names_for_decisions(rows: list[AuditRow], decisions: set[str]) -> list[str]:
    names: list[str] = []
    for row in rows:
        if action_for(row).decision in decisions:
            names.append(row.model.name)
    return names


def display_decision(value: str) -> str:
    if value == "failed_repair":
        return "rejected_repair"
    if value == "reject_host_only":
        return "exclude_host_only"
    return value


def display_reason(value: str) -> str:
    return (
        value.replace("fails strict", "misses strict")
        .replace("fail strict", "miss strict")
        .replace("failed", "rejected")
        .replace("failing", "low-quality")
    )


def markdown(rows: list[AuditRow]) -> str:
    counts = count_decisions(rows)
    release_models = model_names_for_decisions(rows, {"keep_release"})
    candidate_models = model_names_for_decisions(rows, {"keep_candidate"})
    retrain_models = model_names_for_decisions(rows, {"retrain_priority", "retrain_experiment", "retrain_optional"})
    failed_repairs = model_names_for_decisions(rows, {"failed_repair"})
    retired_models = model_names_for_decisions(rows, {"retire_superseded", "reject_host_only"})
    assistants = [row.model.name for row in rows if row.model.role == "assistant_pack"]

    lines = [
        "# GPT2-BASIC Improvement Backlog",
        "",
        "This backlog is evidence-driven. The preview release can iterate now, while low-quality exports stay in the repair or retirement lanes instead of being presented as release options.",
        "",
        "## Current State",
        "",
        f"- Models audited: `{len(rows)}`",
        f"- Strict quality pass: `{pass_count(rows)}/{len(rows)}`",
        f"- Release-ready root models: `{len(release_models)}`",
        f"- Assistant pack models passing: `{len(assistants)}`",
        f"- Priority retrain/experiment queue: `{len(retrain_models)}`",
        f"- Rejected repair attempts recorded as negative evidence: `{len(failed_repairs)}`",
        f"- Retired or host-only exports: `{len(retired_models)}`",
        "",
        "## Iteration Lanes",
        "",
        "| Lane | Immediate goal | Evidence gate |",
        "|---|---|---|",
        "| Preview release | Package only passing release models, assistant packs, source, and selected QEMU evidence. | `python3 scripts/build_preview_release.py --manifest-only` and `--force` for a package tree. |",
        "| Model quality | Retrain 386-min and 486DX2 profiles from the gold curriculum; keep rejected repairs out of release payloads. | `python3 scripts/refresh_model_quality_reports.py` then `python3 scripts/plan_model_quality_repairs.py`. |",
        "| Runtime speed/RAM | Keep measuring full, head-shortlist, q4 token+head, and q4 streamed modes on 486 profiles. | `bash qemu/run_perf_486.sh 486dx2-66 <model-dir>` and `qemu/evidence/hardware_perf_report.md`. |",
        "| Assistant packs | Improve DOSHELP/OFFICE prompt corpora, action routing, and sprite/icon renderer without bloating `GPT2.EXE`. | `bash qemu/run_assistant_486.sh` and `python3 scripts/verify_assistant_packs.py`. |",
        "| Windows/OS2 shells | Reuse `PACK.INI`, `HELP.TXT`, `SPRITE`, `ICONS`, and pack-local model directories in native shells. | Shared pack parser tests plus one DOS, one Windows, and one OS/2 scripted probe. |",
        "| Real hardware | Move the emulator-passing release to one physical 486-class DOS machine first; Pentium data is optional scaling evidence. | `docs/hardware-validation.md` plus real-machine `--quality-all`, `--perf`, and assistant logs. |",
        "",
        "## Release Payload",
        "",
    ]
    for name in release_models:
        row = next(item for item in rows if item.model.name == name)
        lines.append(f"- `{name}`: {quality_text(row)}")

    lines.extend(["", "## Candidate Promotion Queue", ""])
    if candidate_models:
        for name in candidate_models:
            row = next(item for item in rows if item.model.name == name)
            lines.append(f"- `{name}`: {quality_text(row)}; needs DOS vector/perf evidence before release promotion.")
    else:
        lines.append("No host-only candidates are waiting for promotion.")

    lines.extend(["", "## Retrain Queue", ""])
    if retrain_models:
        for name in retrain_models:
            row = next(item for item in rows if item.model.name == name)
            action = action_for(row)
            lines.append(f"- `{name}`: {quality_text(row)}; `{display_decision(action.decision)}`; {display_reason(action.reason)}.")
    else:
        lines.append("No retrain work is currently queued.")

    lines.extend(["", "## Rejected Repairs To Keep Out Of Release", ""])
    if failed_repairs:
        for name in failed_repairs:
            row = next(item for item in rows if item.model.name == name)
            lines.append(f"- `{name}`: {quality_text(row)}")
    else:
        lines.append("No rejected repair artifacts are present.")

    lines.extend(["", "## Decision Counts", ""])
    for decision in sorted(counts):
        lines.append(f"- `{display_decision(decision)}`: {counts[decision]}")

    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "```sh",
            "python3 scripts/build_quality_repair_corpus.py",
            "python3 scripts/build_preview_release.py --manifest-only",
            "python3 scripts/write_improvement_backlog.py",
            "python3 scripts/train_tiny_gpt.py --profile 486dx2-usable --tokenizer lexicon --vocab-size 4096 --lexicon-min-count 1 --base-weight 0 --corpus-file data/domain_curriculum/gold_curriculum_v5_clean_repair.txt --corpus-weight 1 --steps 7000 --log-every 500 --output assets/gpt2_basic/MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR2",
            "python3 scripts/evaluate_gpt2_basic_quality.py --model-dir assets/gpt2_basic/MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR2 --suite all --backend float --threshold 0.72 --output qemu/evidence/quality_report_profile_486dx2_cleanlexicon4096_repair2_all.md",
            "python3 scripts/plan_model_quality_repairs.py",
            "bash qemu/run_assistant_486.sh",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def self_test(rows: list[AuditRow]) -> None:
    text = markdown(rows)
    if "GPT2-BASIC Improvement Backlog" not in text:
        raise RuntimeError("backlog markdown missing title")
    if "Preview release" not in text or "Model quality" not in text:
        raise RuntimeError("backlog missing core lanes")
    print(f"PROBE_OK improvement_backlog_models={len(rows)}")
    print("PROBE_OK improvement_backlog_markdown=1")


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

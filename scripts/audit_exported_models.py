#!/usr/bin/env python3
"""Audit exported GPT2-BASIC model artifacts and quality evidence."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from era_performance_model import Config, parse_config
from profile_pareto_report import (
    QualitySummary,
    heldout_report_path,
    model_key,
    model_suffix,
    parse_quality_report,
    regression_report_path,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = ROOT / "assets" / "gpt2_basic"
DEFAULT_PACK_ROOT = DEFAULT_MODELS_ROOT / "PACKS"
DEFAULT_EVIDENCE = ROOT / "qemu" / "evidence"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "exported_model_quality_inventory.md"

QUALITY_KEY_ALIASES = {
    "MODEL_HEADSHORTLIST2048_PROD_PROBE": ("headshortlist2048",),
}

QUALITY_SUFFIXES = (
    "",
    "_heldout",
    "_runtime",
    "_all",
    "_fixed_all",
    "_all_stopfix",
    "_heldout_stopfix",
    "_runtime_stopfix",
    "_heldout_samplerfix",
    "_runtime_samplerfix",
)


@dataclass(frozen=True)
class ExportedModel:
    name: str
    role: str
    path: Path


@dataclass(frozen=True)
class AuditRow:
    model: ExportedModel
    cfg: Config | None
    artifact_ok: bool
    artifact_log: Path | None
    quality_reports: tuple[QualitySummary, ...]


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def find_root_models(models_root: Path) -> list[ExportedModel]:
    models: list[ExportedModel] = []
    for path in sorted(models_root.glob("MODEL*")):
        if path.is_dir() and (path / "GPT2CFG.TXT").exists():
            models.append(ExportedModel(path.name, "root", path))
    return models


def pack_ids(pack_root: Path) -> list[str]:
    list_path = pack_root / "PACKS.TXT"
    if not list_path.exists():
        return []
    ids: list[str] = []
    for raw in list_path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and not line.startswith(";"):
            ids.append(line.upper())
    return ids


def find_assistant_models(pack_root: Path) -> list[ExportedModel]:
    models: list[ExportedModel] = []
    for pack_id in pack_ids(pack_root):
        path = pack_root / pack_id / "MODEL"
        if path.is_dir() and (path / "GPT2CFG.TXT").exists():
            models.append(ExportedModel(f"ASSISTANT_{pack_id}", "assistant_pack", path))
    return models


def shape_text(cfg: Config | None) -> str:
    if cfg is None:
        return "unknown"
    return f"{cfg.n_layer}L {cfg.n_embd}D {cfg.n_head}H ctx{cfg.n_positions} h{cfg.hidden_dim} v{cfg.vocab_size}"


def report_matches(report: QualitySummary, seen: set[Path]) -> bool:
    if report.path in seen:
        return False
    seen.add(report.path)
    return True


def root_quality_reports(model_dir: Path, cfg: Config, evidence_dir: Path) -> tuple[QualitySummary, ...]:
    candidates: list[Path] = [
        regression_report_path(evidence_dir, model_dir),
        heldout_report_path(evidence_dir, model_dir),
        model_dir / "quality_heldout.md",
        model_dir / "quality_runtime.md",
    ]
    keys = (model_key(model_dir),) + QUALITY_KEY_ALIASES.get(model_dir.name, ())
    suffix = model_suffix(model_dir)
    if model_dir.name == "MODEL":
        candidates.extend(
            [
                evidence_dir / "quality_report.md",
                evidence_dir / "quality_report_heldout.md",
                evidence_dir / "quality_report_default_model_all.md",
                evidence_dir / "quality_report_default_model_fixed_all.md",
                evidence_dir / "quality_report_dos.md",
                evidence_dir / "quality_report_dos_heldout.md",
                evidence_dir / "quality_report_dos_all.md",
            ]
        )
    else:
        for key in keys:
            candidates.extend(evidence_dir / f"quality_report_{key}{report_suffix}.md" for report_suffix in QUALITY_SUFFIXES)
        candidates.extend(evidence_dir / f"quality_report_dos{suffix}{report_suffix}.md" for report_suffix in QUALITY_SUFFIXES)

    reports: list[QualitySummary] = []
    seen: set[Path] = set()
    for path in candidates:
        report = parse_quality_report(path, cfg)
        if report is not None and report_matches(report, seen):
            reports.append(report)
    return tuple(sorted(reports, key=lambda item: (item.status != "PASS", -item.average, item.path.name)))


def assistant_quality_reports(model: ExportedModel, cfg: Config, evidence_dir: Path) -> tuple[QualitySummary, ...]:
    pack_id = model.name.removeprefix("ASSISTANT_").lower()
    path = evidence_dir / f"quality_report_assistant_{pack_id}.md"
    report = parse_quality_report(path, cfg)
    return (report,) if report is not None else ()


def quality_reports_for(model: ExportedModel, cfg: Config, evidence_dir: Path) -> tuple[QualitySummary, ...]:
    if model.role == "assistant_pack":
        return assistant_quality_reports(model, cfg, evidence_dir)
    return root_quality_reports(model.path, cfg, evidence_dir)


def best_quality(reports: tuple[QualitySummary, ...]) -> QualitySummary | None:
    if not reports:
        return None
    passes = [report for report in reports if report.status == "PASS" and report.passed == report.total]
    if passes:
        return max(passes, key=lambda report: report.average)
    return max(reports, key=lambda report: report.average)


def required_quality(reports: tuple[QualitySummary, ...]) -> QualitySummary | None:
    all_suite = [report for report in reports if report.suite == "all"]
    if all_suite:
        return max(
            all_suite,
            key=lambda report: (
                report.status == "PASS" and report.passed == report.total,
                report.passed / max(1, report.total),
                report.average,
            ),
        )
    return best_quality(reports)


def quality_cell(reports: tuple[QualitySummary, ...]) -> str:
    report = required_quality(reports)
    if report is None:
        return "missing"
    return f"{report.status} {report.passed}/{report.total} avg {report.average:.3f} ({report.backend}, {report.suite})"


def run_model_report(model: ExportedModel, evidence_dir: Path) -> tuple[bool, Path]:
    log_path = evidence_dir / f"model_inventory_{safe_name(model.name)}.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "model_report.py"),
        "--model-dir",
        str(model.path),
        "--strict",
    ]
    with log_path.open("w", encoding="ascii", errors="ignore") as log_file:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="ascii",
            errors="ignore",
            check=False,
        )
    return process.returncode == 0, log_path


def audit_model(model: ExportedModel, evidence_dir: Path, refresh_model_reports: bool) -> AuditRow:
    cfg: Config | None = None
    artifact_ok = False
    artifact_log: Path | None = None
    try:
        cfg = parse_config(model.path / "GPT2CFG.TXT")
    except Exception:
        cfg = None

    if refresh_model_reports:
        artifact_ok, artifact_log = run_model_report(model, evidence_dir)
    else:
        required = ("GPT2CFG.TXT", "GPT2WT.BIN", "GPT2FX.BIN", "GPT2EXP.BIN", "PROFILE.TXT")
        artifact_ok = all((model.path / name).exists() for name in required)

    reports = quality_reports_for(model, cfg, evidence_dir) if cfg is not None else ()
    return AuditRow(model, cfg, artifact_ok, artifact_log, reports)


def audit_models(
    models_root: Path,
    pack_root: Path,
    evidence_dir: Path,
    refresh_model_reports: bool,
) -> list[AuditRow]:
    models = find_root_models(models_root) + find_assistant_models(pack_root)
    return [audit_model(model, evidence_dir, refresh_model_reports) for model in models]


def report_paths(reports: tuple[QualitySummary, ...]) -> str:
    if not reports:
        return "none"
    names = [report.path.name for report in reports[:3]]
    if len(reports) > 3:
        names.append(f"+{len(reports) - 3} more")
    return ", ".join(f"`{name}`" for name in names)


def markdown(rows: list[AuditRow], evidence_dir: Path) -> str:
    total = len(rows)
    artifact_pass = sum(1 for row in rows if row.artifact_ok)
    quality_pass = sum(
        1 for row in rows if (required_quality(row.quality_reports) or None) and required_quality(row.quality_reports).status == "PASS"
    )
    quality_missing = sum(1 for row in rows if not row.quality_reports)
    needs_training = total - quality_pass - quality_missing

    lines = [
        "# Exported Model Quality Inventory",
        "",
        f"Models audited: `{total}`",
        f"Artifact pass: `{artifact_pass}/{total}`",
        f"Quality pass: `{quality_pass}/{total}`",
        f"Needs training: `{needs_training}`",
        f"Missing quality evidence: `{quality_missing}`",
        "",
        "## Models",
        "",
        "| Role | Model | Shape | Artifacts | Best Quality | Reports |",
        "|---|---|---|---|---|---|",
    ]

    for row in sorted(rows, key=lambda item: (item.model.role, item.model.name)):
        artifact_text = "PASS" if row.artifact_ok else "FAIL"
        if row.artifact_log is not None:
            artifact_text += f" `{row.artifact_log.relative_to(ROOT)}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model.role,
                    f"`{row.model.name}`",
                    f"`{shape_text(row.cfg)}`",
                    artifact_text,
                    quality_cell(row.quality_reports),
                    report_paths(row.quality_reports),
                ]
            )
            + " |"
        )

    failing = [
        row
        for row in rows
        if not row.artifact_ok or not row.quality_reports or (required_quality(row.quality_reports) or None).status != "PASS"
    ]
    lines.extend(["", "## Gaps", ""])
    if not failing:
        lines.append("No model inventory gaps found.")
    else:
        for row in failing:
            reasons: list[str] = []
            if not row.artifact_ok:
                reasons.append("artifact validation")
            if not row.quality_reports:
                reasons.append("missing quality")
            elif required_quality(row.quality_reports) is not None and required_quality(row.quality_reports).status != "PASS":
                reasons.append("quality needs training")
            lines.append(f"- `{row.model.name}`: {', '.join(reasons)}")

    lines.extend(["", f"Evidence root: `{evidence_dir}`", ""])
    return "\n".join(lines)


def self_test(models_root: Path, pack_root: Path, evidence_dir: Path) -> None:
    rows = audit_models(models_root, pack_root, evidence_dir, refresh_model_reports=False)
    if not rows:
        raise RuntimeError("no exported models found")
    text = markdown(rows[:2], evidence_dir)
    if "Exported Model Quality Inventory" not in text:
        raise RuntimeError("inventory markdown missing title")
    print(f"PROBE_OK exported_model_inventory_count={len(rows)}")
    print("PROBE_OK exported_model_inventory_markdown=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-model-reports", action="store_true")
    parser.add_argument("--require-quality", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.models_root, args.pack_root, args.evidence_dir)
        return

    rows = audit_models(args.models_root, args.pack_root, args.evidence_dir, args.refresh_model_reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(rows, args.evidence_dir) + "\n", encoding="ascii")
    print(f"wrote {args.output}")

    if args.require_quality:
        missing_or_failing = [
            row.model.name
            for row in rows
            if not row.quality_reports or (required_quality(row.quality_reports) or None).status != "PASS"
        ]
        if missing_or_failing:
            raise SystemExit("MODEL_QUALITY_INVENTORY_FAILED " + ",".join(missing_or_failing))


if __name__ == "__main__":
    main()

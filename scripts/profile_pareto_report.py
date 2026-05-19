#!/usr/bin/env python3
"""Build a GPT2-BASIC checkpoint/profile Pareto report."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from era_performance_model import (
    DEFAULT_PERF_LOGS,
    MACHINES,
    Config,
    estimate_runtime_memory,
    parameter_count,
    parse_config,
    parse_perf_log,
    seconds_for,
    work_counts,
)
from evaluate_gpt2_basic_quality import (
    DEFAULT_MIN_GENERATED,
    evaluate_model,
    markdown_report,
    quality_prompts,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = ROOT / "assets" / "gpt2_basic"
DEFAULT_EVIDENCE = ROOT / "qemu" / "evidence"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "profile_pareto_report.md"


@dataclass(frozen=True)
class QualitySummary:
    path: Path
    suite: str
    backend: str
    status: str
    average: float
    passed: int
    total: int


@dataclass(frozen=True)
class ProfileRow:
    model_dir: Path
    cfg: Config
    params: int
    runtime_bytes: int
    estimated_486dx2_tps: float
    measured_486dx2_tps: float | None
    measured_host_tps: float | None
    regression: QualitySummary | None
    heldout: QualitySummary | None


QUALITY_REPORT_BY_MODEL = {
    "MODEL": "quality_report.md",
    "MODEL_CANDIDATE": "quality_report_candidate.md",
    "MODEL_CANDIDATE_FINETUNE1": "quality_report_candidate_finetune1.md",
    "MODEL_CANDIDATE_MPS": "quality_report_candidate_mps.md",
    "MODEL_CANDIDATE_MPS2": "quality_report_candidate_mps2.md",
    "MODEL_CANDIDATE_MPS3": "quality_report_candidate_mps3.md",
}


def model_key(model_dir: Path) -> str:
    return model_dir.name.lower().removeprefix("model_").replace("model", "current")


def model_suffix(model_dir: Path) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", model_dir.name.lower()).strip("_")
    if key == "model":
        return ""
    return "_" + key


def dos_heldout_report_path(evidence_dir: Path, model_dir: Path) -> Path:
    return evidence_dir / f"quality_report_dos{model_suffix(model_dir)}_heldout.md"


def heldout_report_path(evidence_dir: Path, model_dir: Path) -> Path:
    dos_path = dos_heldout_report_path(evidence_dir, model_dir)
    if dos_path.exists():
        return dos_path
    if model_dir.name == "MODEL":
        return evidence_dir / "quality_report_heldout.md"
    return evidence_dir / f"quality_report_{model_key(model_dir)}_heldout.md"


def heldout_float_report_path(evidence_dir: Path, model_dir: Path) -> Path:
    if model_dir.name == "MODEL":
        return evidence_dir / "quality_report_heldout.md"
    return evidence_dir / f"quality_report_{model_key(model_dir)}_heldout.md"


def quality_report_matches_config(text: str, cfg: Config) -> bool:
    shape_match = re.search(
        r"Shape:\s+`(\d+)L\s+(\d+)D\s+(\d+)H\s+ctx(\d+)\s+hidden(\d+)\s+vocab(\d+)`",
        text,
    )
    if shape_match is None:
        return True
    values = tuple(int(value) for value in shape_match.groups())
    expected = (cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_positions, cfg.hidden_dim, cfg.vocab_size)
    return values == expected


def parse_quality_report(path: Path, cfg: Config | None = None) -> QualitySummary | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="ascii", errors="ignore")
    if cfg is not None and not quality_report_matches_config(text, cfg):
        return None

    suite_match = re.search(r"Quality suite:\s+`([^`]+)`", text)
    backend_match = re.search(r"Evaluation backend:\s+`([^`]+)`", text)
    status_match = re.search(r"Quality status:\s+`([^`]+)`", text)
    average_match = re.search(r"Average score:\s+`([0-9.]+)`", text)
    pass_match = re.search(r"Prompt pass rate:\s+`(\d+)/(\d+)`", text)

    if status_match is None or average_match is None or pass_match is None:
        return None

    return QualitySummary(
        path=path,
        suite=suite_match.group(1) if suite_match else "runtime-regression",
        backend=backend_match.group(1) if backend_match else "unknown",
        status=status_match.group(1),
        average=float(average_match.group(1)),
        passed=int(pass_match.group(1)),
        total=int(pass_match.group(2)),
    )


def fmt_quality(summary: QualitySummary | None) -> str:
    if summary is None:
        return "missing"
    return f"{summary.status} {summary.passed}/{summary.total} avg {summary.average:.3f} ({summary.backend})"


def find_model_dirs(root: Path) -> list[Path]:
    dirs = []
    for path in sorted(root.glob("MODEL*")):
        if path.is_dir() and (path / "GPT2CFG.TXT").exists():
            dirs.append(path)
    return dirs


def refresh_heldout_reports(model_dirs: list[Path], evidence_dir: Path) -> None:
    prompts = quality_prompts("heldout")
    for model_dir in model_dirs:
        cfg, results = evaluate_model(
            model_dir,
            prompts,
            max_new_tokens=90,
            min_generated=DEFAULT_MIN_GENERATED,
            threshold=0.72,
            backend="float",
            device_name="cpu",
        )
        report = markdown_report(cfg, results, 0.72, "float", "heldout")
        out = heldout_float_report_path(evidence_dir, model_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="ascii")
        print(f"wrote {out}")


def perf_log_matches_config(path: Path, cfg: Config, runtime_bytes: int) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="ascii", errors="ignore")
    model_match = re.search(r"^PERF_MODEL\|(.+)$", text, flags=re.MULTILINE)
    if model_match is None:
        return False
    fields = dict(part.split("=", 1) for part in model_match.group(1).split("|") if "=" in part)
    expected = {
        "layers": cfg.n_layer,
        "emb": cfg.n_embd,
        "heads": cfg.n_head,
        "ctx": cfg.n_positions,
        "vocab": cfg.vocab_size,
        "params": parameter_count(cfg),
        "runtime_bytes": runtime_bytes,
    }
    for key, value in expected.items():
        try:
            if int(fields.get(key, "")) != value:
                return False
        except ValueError:
            return False
    return True


def matching_perf_log(path: Path, cfg: Config, runtime_bytes: int):
    if not perf_log_matches_config(path, cfg, runtime_bytes):
        return None
    return parse_perf_log(path)


def measured_tps_by_profile(perf_logs: list[Path], cfg: Config, runtime_bytes: int) -> tuple[float | None, float | None]:
    measured_486dx2 = None
    measured_host = None
    for path in perf_logs:
        run = matching_perf_log(path, cfg, runtime_bytes)
        if run is None:
            continue
        if run.profile == "486dx2-66":
            measured_486dx2 = run.tokens_per_sec
        elif run.profile == "host":
            measured_host = run.tokens_per_sec
    return measured_486dx2, measured_host


def measured_tps_for_model(
    evidence_dir: Path,
    model_dir: Path,
    perf_logs: list[Path],
    cfg: Config,
    runtime_bytes: int,
) -> tuple[float | None, float | None]:
    suffix = model_suffix(model_dir)
    if suffix:
        model_486 = matching_perf_log(evidence_dir / f"perf_486_486dx2-66{suffix}.log", cfg, runtime_bytes)
        model_host = matching_perf_log(evidence_dir / f"perf_486_host{suffix}.log", cfg, runtime_bytes)
        return (
            model_486.tokens_per_sec if model_486 is not None else None,
            model_host.tokens_per_sec if model_host is not None else None,
        )
    return measured_tps_by_profile(perf_logs, cfg, runtime_bytes)


def regression_report_path(evidence_dir: Path, model_dir: Path) -> Path:
    dos_path = evidence_dir / f"quality_report_dos{model_suffix(model_dir)}.md"
    if dos_path.exists():
        return dos_path
    regression_name = QUALITY_REPORT_BY_MODEL.get(model_dir.name)
    if regression_name:
        return evidence_dir / regression_name
    return dos_path


def estimated_486dx2_tps(cfg: Config) -> float:
    counts = work_counts(cfg, prompt_tokens=31, generated_tokens=70)
    for machine in MACHINES:
        if machine.key == "486dx2-66":
            return 70.0 / seconds_for(machine, counts["weighted_after"])
    raise RuntimeError("missing 486dx2-66 machine profile")


def build_rows(model_dirs: list[Path], evidence_dir: Path, perf_logs: list[Path]) -> list[ProfileRow]:
    rows: list[ProfileRow] = []

    for model_dir in model_dirs:
        cfg = parse_config(model_dir / "GPT2CFG.TXT")
        runtime_bytes = estimate_runtime_memory(cfg)
        measured_486dx2, measured_host = measured_tps_for_model(evidence_dir, model_dir, perf_logs, cfg, runtime_bytes)
        regression = parse_quality_report(regression_report_path(evidence_dir, model_dir), cfg)
        heldout = parse_quality_report(heldout_report_path(evidence_dir, model_dir), cfg)

        rows.append(
            ProfileRow(
                model_dir=model_dir,
                cfg=cfg,
                params=parameter_count(cfg),
                runtime_bytes=runtime_bytes,
                estimated_486dx2_tps=estimated_486dx2_tps(cfg),
                measured_486dx2_tps=measured_486dx2,
                measured_host_tps=measured_host,
                regression=regression,
                heldout=heldout,
            )
        )

    return rows


def score_row(row: ProfileRow) -> float:
    heldout = row.heldout.average if row.heldout is not None else 0.0
    tps = row.measured_486dx2_tps or row.estimated_486dx2_tps
    memory_mb = row.runtime_bytes / (1024.0 * 1024.0)
    return heldout * tps / max(0.1, memory_mb)


def markdown(rows: list[ProfileRow], evidence_dir: Path) -> str:
    ranked = sorted(rows, key=score_row, reverse=True)
    shape_counts: dict[str, int] = {}
    for row in rows:
        shape = f"{row.cfg.n_layer}L {row.cfg.n_embd}D {row.cfg.n_head}H ctx{row.cfg.n_positions} h{row.cfg.hidden_dim} v{row.cfg.vocab_size}"
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    lines = [
        "# GPT2-BASIC Profile Pareto Report",
        "",
        "This report ranks exported checkpoints using shape-derived memory/work estimates, available QEMU `--perf` measurements, and held-out quality evidence.",
        "",
        "QEMU speed is per-model when a profile-specific `--perf` log exists; otherwise the row uses the shape-derived model estimate. It is still emulator evidence, not physical-board timing.",
        "",
        "## Ranking",
        "",
        "| Rank | Model dir | Shape | Params | Runtime bytes | 486DX2 tok/s | Host tok/s | Regression quality | Held-out quality | Score |",
        "|---:|---|---|---:|---:|---:|---:|---|---|---:|",
    ]

    for rank, row in enumerate(ranked, start=1):
        measured_486 = row.measured_486dx2_tps
        measured_host = row.measured_host_tps
        shape = f"{row.cfg.n_layer}L {row.cfg.n_embd}D {row.cfg.n_head}H ctx{row.cfg.n_positions} h{row.cfg.hidden_dim} v{row.cfg.vocab_size}"
        tps_text = f"{measured_486:.2f} measured" if measured_486 is not None else f"{row.estimated_486dx2_tps:.2f} model"
        host_text = f"{measured_host:.2f}" if measured_host is not None else "missing"
        lines.append(
            f"| {rank} | `{row.model_dir.name}` | `{shape}` | {row.params} | {row.runtime_bytes} | {tps_text} | {host_text} | "
            f"{fmt_quality(row.regression)} | {fmt_quality(row.heldout)} | {score_row(row):.2f} |"
        )

    missing_heldout = [row.model_dir.name for row in rows if row.heldout is None]
    missing_regression = [row.model_dir.name for row in rows if row.regression is None]
    missing_perf = [row.model_dir.name for row in rows if row.measured_486dx2_tps is None]
    active_model = next((row for row in rows if row.model_dir.name == "MODEL"), None)

    lines.extend(
        [
            "",
            "## Shape Coverage",
            "",
            "| Shape | Checkpoints |",
            "|---|---:|",
        ]
    )
    for shape, count in sorted(shape_counts.items()):
        lines.append(f"| `{shape}` | {count} |")

    lines.append("")
    if len(shape_counts) > 1:
        lines.append(
            "Current exports now cover multiple shapes. Use `architecture_profile_sweep.py` for the remaining trainer profiles and profile-level DOS evidence."
        )
    else:
        lines.append(
            "Current exported checkpoints do not yet exercise the larger or smaller architecture profiles exposed by the trainer."
        )
    if active_model is not None and active_model.heldout is not None:
        lines.append(
            f"Active `MODEL` held-out source: `{active_model.heldout.path.name}` "
            f"({active_model.heldout.backend})."
        )

    lines.extend(
        [
            "",
            "## Gaps",
            "",
            f"- Missing held-out quality reports: {', '.join(missing_heldout) if missing_heldout else 'none'}",
            f"- Missing runtime-regression quality reports: {', '.join(missing_regression) if missing_regression else 'none'}",
            f"- Missing direct DOS `--perf` per checkpoint: {', '.join(missing_perf) if missing_perf else 'none'}",
            "",
            "## Next Commands",
            "",
            "```sh",
            "python3 scripts/profile_pareto_report.py --refresh-heldout-float",
            "bash qemu/run_perf_486.sh 386dx-33",
            "bash qemu/run_perf_486.sh 486sx-25",
            "bash qemu/run_perf_486.sh 486dx-33",
            "bash qemu/run_perf_486.sh 486dx4-100",
            "bash qemu/run_perf_486.sh pentium-60",
            "bash qemu/run_perf_486.sh pentium-133",
            "```",
            "",
            f"Evidence root: `{evidence_dir}`",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test(models_root: Path, evidence_dir: Path) -> None:
    model_dirs = find_model_dirs(models_root)
    if not model_dirs:
        raise RuntimeError("no model dirs found")
    rows = build_rows(model_dirs[:1], evidence_dir, DEFAULT_PERF_LOGS)
    report = markdown(rows, evidence_dir)
    measured_tps_by_profile(DEFAULT_PERF_LOGS, rows[0].cfg, rows[0].runtime_bytes)
    measured_tps_for_model(evidence_dir, model_dirs[0], DEFAULT_PERF_LOGS, rows[0].cfg, rows[0].runtime_bytes)
    estimated = estimated_486dx2_tps(rows[0].cfg)
    print("trace parse_quality_report")
    print("trace estimated_486dx2_tps")
    print("trace measured_tps_by_profile")
    print("trace measured_tps_for_model")
    print(f"PROBE_OK estimated_486dx2_tps value={estimated:.3f}")
    print(f"PROBE_OK find_model_dirs count={len(model_dirs)}")
    print(f"PROBE_OK build_rows count={len(rows)}")
    print(f"PROBE_OK markdown bytes={len(report)}")
    print("PROBE_OK profile_pareto_cli self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--perf-log", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-heldout-float", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.models_root, args.evidence_dir)
        return

    model_dirs = find_model_dirs(args.models_root)
    if args.refresh_heldout_float:
        refresh_heldout_reports(model_dirs, args.evidence_dir)

    perf_logs = args.perf_log if args.perf_log is not None else DEFAULT_PERF_LOGS
    rows = build_rows(model_dirs, args.evidence_dir, perf_logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(rows, args.evidence_dir), encoding="ascii")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

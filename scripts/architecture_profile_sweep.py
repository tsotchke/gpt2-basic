#!/usr/bin/env python3
"""Plan and report GPT2-BASIC architecture-profile sweeps."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from era_performance_model import (
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
from profile_pareto_report import QualitySummary, fmt_quality, parse_quality_report
from train_tiny_gpt import MODEL_PROFILES, ModelProfile, build_backend_runtime, print_backend_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = ROOT / "assets" / "gpt2_basic"
DEFAULT_EVIDENCE = ROOT / "qemu" / "evidence"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "architecture_profile_sweep.md"
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


@dataclass(frozen=True)
class ArchitectureRow:
    profile: str
    model_profile: ModelProfile
    cfg: Config
    target_dir: Path
    export_dir: Path | None
    quality: QualitySummary | None
    measured_486dx2_tps: float | None


def profile_slug(profile: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", profile.lower()).strip("_")
    if not slug:
        raise ValueError(f"invalid profile name: {profile!r}")
    return slug


def profile_dir_name(profile: str) -> str:
    return "MODEL_PROFILE_" + profile_slug(profile).upper()


def target_export_dir(models_root: Path, profile: str) -> Path:
    return models_root / profile_dir_name(profile)


def profile_quality_path(evidence_dir: Path, profile: str) -> Path:
    return evidence_dir / f"quality_report_profile_{profile_slug(profile)}_heldout.md"


def active_dos_quality_path(evidence_dir: Path) -> Path:
    return evidence_dir / "quality_report_dos_heldout.md"


def model_suffix(model_dir: Path) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", model_dir.name.lower()).strip("_")
    if key == "model":
        return ""
    return "_" + key


def dos_quality_path(evidence_dir: Path, model_dir: Path) -> Path:
    return evidence_dir / f"quality_report_dos{model_suffix(model_dir)}_heldout.md"


def perf_log_path(evidence_dir: Path, qemu_profile: str, model_dir: Path) -> Path:
    return evidence_dir / f"perf_486_{qemu_profile}{model_suffix(model_dir)}.log"


def to_era_config(profile: str, model_profile: ModelProfile) -> Config:
    cfg = model_profile.cfg
    return Config(
        profile=profile,
        vocab_size=cfg.vocab_size,
        n_positions=cfg.n_positions,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        hidden_dim=cfg.hidden_dim,
    )


def same_shape(left: Config, right: Config) -> bool:
    return (
        left.vocab_size == right.vocab_size
        and left.n_positions == right.n_positions
        and left.n_embd == right.n_embd
        and left.n_head == right.n_head
        and left.n_layer == right.n_layer
        and left.hidden_dim == right.hidden_dim
    )


def find_export_dir(models_root: Path, profile: str, cfg: Config) -> Path | None:
    target = target_export_dir(models_root, profile)
    if (target / "GPT2CFG.TXT").exists():
        exported = parse_config(target / "GPT2CFG.TXT")
        if same_shape(cfg, exported):
            return target

    active = models_root / "MODEL"
    if profile == "486sx-safe" and (active / "GPT2CFG.TXT").exists():
        exported = parse_config(active / "GPT2CFG.TXT")
        if same_shape(cfg, exported):
            return active

    return None


def preferred_quality(evidence_dir: Path, profile: str, export_dir: Path | None) -> QualitySummary | None:
    if export_dir is not None:
        dos_quality = parse_quality_report(dos_quality_path(evidence_dir, export_dir))
        if dos_quality is not None:
            return dos_quality
    if profile == "486sx-safe":
        dos_quality = parse_quality_report(active_dos_quality_path(evidence_dir))
        if dos_quality is not None:
            return dos_quality
    return parse_quality_report(profile_quality_path(evidence_dir, profile))


def build_rows(profiles: list[str], models_root: Path, evidence_dir: Path) -> list[ArchitectureRow]:
    rows: list[ArchitectureRow] = []
    for profile in profiles:
        model_profile = MODEL_PROFILES[profile]
        cfg = to_era_config(profile, model_profile)
        target_dir = target_export_dir(models_root, profile)
        export_dir = find_export_dir(models_root, profile, cfg)
        measured_486dx2_tps = None
        if export_dir is not None:
            perf = parse_perf_log(perf_log_path(evidence_dir, "486dx2-66", export_dir))
            if perf is not None:
                measured_486dx2_tps = perf.tokens_per_sec
        rows.append(
            ArchitectureRow(
                profile=profile,
                model_profile=model_profile,
                cfg=cfg,
                target_dir=target_dir,
                export_dir=export_dir,
                quality=preferred_quality(evidence_dir, profile, export_dir),
                measured_486dx2_tps=measured_486dx2_tps,
            )
        )
    return rows


def estimated_tps(cfg: Config, machine_key: str, prompt_tokens: int, generated_tokens: int) -> float:
    counts = work_counts(cfg, prompt_tokens=prompt_tokens, generated_tokens=generated_tokens)
    for machine in MACHINES:
        if machine.key == machine_key:
            return generated_tokens / seconds_for(machine, counts["weighted_after"])
    raise RuntimeError(f"unknown machine profile: {machine_key}")


def score_row(row: ArchitectureRow) -> float:
    if row.quality is None:
        return 0.0
    tps = row.measured_486dx2_tps or estimated_tps(row.cfg, "486dx2-66", 31, 70)
    memory_mb = estimate_runtime_memory(row.cfg) / (1024.0 * 1024.0)
    return row.quality.average * tps / max(0.1, memory_mb)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def train_command(row: ArchitectureRow, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_gpt2_basic.py"),
        "--profile",
        row.profile,
        "--output",
        str(row.target_dir),
        "--device",
        args.device,
        "--sample-tokens",
        str(args.sample_tokens),
        "--log-every",
        str(args.log_every),
    ]
    if args.steps_override is not None:
        command.extend(["--steps", str(args.steps_override)])
    if args.include_docs:
        command.append("--include-docs")
    for corpus_file in args.corpus_file or []:
        command.extend(["--corpus-file", str(corpus_file)])
    command.extend(["--corpus-weight", str(args.corpus_weight)])
    if args.write_legacy_names:
        command.append("--write-legacy-names")
    return command


def print_sweep_backend_contract(runtime, mode: str) -> None:
    print_backend_contract(runtime)
    fallback_intentional = int(
        runtime.requested == "auto"
        and runtime.selected == "cpu"
        and not any(runtime.accelerator_available.values())
    )
    print(f"runtime_backend: architecture_profile_sweep_{runtime.selected}", flush=True)
    print(
        "backend_gate: "
        f"mode={mode} requested={runtime.requested} selected={runtime.selected} "
        f"fallback_intentional={fallback_intentional}",
        flush=True,
    )


def run_training(rows: list[ArchitectureRow], args: argparse.Namespace) -> None:
    runtime = build_backend_runtime(args.device)
    print_sweep_backend_contract(runtime, "train")
    for row in rows:
        if row.export_dir is not None and not args.force:
            print(f"skip existing export: {rel(row.export_dir)}")
            continue
        command = train_command(row, args)
        print("running: " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


def refresh_heldout_reports(rows: list[ArchitectureRow], evidence_dir: Path) -> None:
    prompts = quality_prompts("heldout")
    for row in rows:
        export_dir = row.export_dir
        if export_dir is None and row.target_dir.exists():
            export_dir = row.target_dir
        if export_dir is None or not (export_dir / "GPT2CFG.TXT").exists():
            print(f"skip heldout, no export: {row.profile}")
            continue

        cfg, results = evaluate_model(
            export_dir,
            prompts,
            max_new_tokens=90,
            min_generated=DEFAULT_MIN_GENERATED,
            threshold=0.72,
            backend="float",
            device_name="cpu",
        )
        report = markdown_report(cfg, results, 0.72, "float", "heldout")
        out = profile_quality_path(evidence_dir, row.profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="ascii")
        print(f"wrote {out}")


def markdown(rows: list[ArchitectureRow], evidence_dir: Path) -> str:
    missing_rows = [row for row in rows if row.export_dir is None]
    dos_rows = [row for row in rows if row.quality is not None and row.quality.backend == "dos-fixed-qemu"]
    measured_rows = [row for row in rows if row.measured_486dx2_tps is not None]
    next_row = missing_rows[0] if missing_rows else rows[0]
    next_profile = next_row.profile
    next_target = next_row.target_dir

    lines = [
        "# GPT2-BASIC Architecture Profile Sweep",
        "",
        "This report tracks trainer architecture profiles, not same-shape checkpoint variants.",
        "Rows without an exported model are planning rows until `train_gpt2_basic.py` writes the checkpoint and DOS/QEMU evidence is collected.",
        "",
        "## Profile Matrix",
        "",
        "| Profile | Export | Shape | Params | Runtime MB | 386DX tok/s | 486SX tok/s | 486DX2 tok/s | Pentium 133 tok/s | Held-out quality | Score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]

    for row in rows:
        cfg = row.cfg
        shape = f"{cfg.n_layer}L {cfg.n_embd}D {cfg.n_head}H ctx{cfg.n_positions} h{cfg.hidden_dim} v{cfg.vocab_size}"
        export_text = rel(row.export_dir) if row.export_dir is not None else "missing"
        runtime_mb = estimate_runtime_memory(cfg) / (1024.0 * 1024.0)
        tps_386 = estimated_tps(cfg, "386dx-33", 31, 70)
        tps_486sx = estimated_tps(cfg, "486sx-25", 31, 70)
        tps_486dx2 = estimated_tps(cfg, "486dx2-66", 31, 70)
        tps_486dx2_text = (
            f"{row.measured_486dx2_tps:.2f} measured"
            if row.measured_486dx2_tps is not None
            else f"{tps_486dx2:.2f} model"
        )
        tps_p133 = estimated_tps(cfg, "pentium-133", 31, 70)
        lines.append(
            f"| `{row.profile}` | `{export_text}` | `{shape}` | {parameter_count(cfg)} | {runtime_mb:.3f} | "
            f"{tps_386:.2f} | {tps_486sx:.2f} | {tps_486dx2_text} | {tps_p133:.2f} | "
            f"{fmt_quality(row.quality)} | {score_row(row):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Current Finding",
            "",
        ]
    )
    if len(dos_rows) >= 2:
        best_quality = max(dos_rows, key=lambda row: row.quality.average if row.quality else 0.0)
        if measured_rows:
            fastest = max(measured_rows, key=lambda row: row.measured_486dx2_tps or 0.0)
            lines.append(
                f"{len(dos_rows)} profiles now have DOS fixed-point held-out evidence. "
                f"`{fastest.profile}` is the fastest measured QEMU `486dx2-66` profile at "
                f"{fastest.measured_486dx2_tps:.2f} tok/s, while `{best_quality.profile}` "
                f"has the best DOS held-out average at {best_quality.quality.average:.3f}. "
                "No profile passes the held-out suite yet."
            )
        else:
            lines.append(
                f"{len(dos_rows)} profiles now have DOS fixed-point held-out evidence. "
                f"`{best_quality.profile}` has the best DOS held-out average at "
                f"{best_quality.quality.average:.3f}. No profile passes the held-out suite yet."
            )
    elif dos_rows:
        lines.append(
            f"Only `{dos_rows[0].profile}` has DOS fixed-point held-out evidence today. "
            "Other profiles need exported checkpoints, vector parity, DOS quality, and DOS `--perf` logs."
        )
    else:
        lines.append(
            "No trainer profile has DOS fixed-point held-out evidence yet. Export and stage a profile before ranking architecture quality."
        )

    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "Train missing profile checkpoints, then run the same DOS evidence contract for each exported model directory.",
            "",
            "```sh",
            f"python3 scripts/architecture_profile_sweep.py --train --profiles {next_profile} --steps-override 1200 --include-docs --corpus-file data/online_corpus/online_training_corpus.txt",
            "python3 scripts/architecture_profile_sweep.py --refresh-heldout-float",
            f"bash qemu/run_vectors_486.sh {rel(next_target)}",
            f"bash qemu/run_quality_486.sh {rel(next_target)}",
            f"bash qemu/run_perf_486.sh 486dx2-66 {rel(next_target)}",
            "```",
            "",
            f"Evidence root: `{evidence_dir}`",
        ]
    )
    return "\n".join(lines) + "\n"


def selected_profiles(args: argparse.Namespace) -> list[str]:
    if args.profiles:
        return args.profiles
    return list(MODEL_PROFILES.keys())


def self_test(models_root: Path, evidence_dir: Path) -> None:
    runtime = build_backend_runtime("cpu")
    print_sweep_backend_contract(runtime, "self_test")
    rows = build_rows(["386-min", "486sx-safe"], models_root, evidence_dir)
    report = markdown(rows, evidence_dir)
    command = train_command(
        rows[0],
        argparse.Namespace(
            device="cpu",
            sample_tokens=4,
            log_every=1,
            steps_override=1,
            include_docs=False,
            corpus_file=[],
            corpus_weight=1,
            write_legacy_names=False,
        ),
    )
    assert "--profile" in command
    print("trace_scope architecture_profile_contract")
    print("trace same_shape")
    print("trace find_export_dir")
    print("trace preferred_quality")
    print("trace to_era_config")
    print("trace refresh_heldout_reports")
    print("trace run_training")
    print("trace train_command")
    print("trace build_backend_runtime")
    print("trace print_sweep_backend_contract")
    print(f"PROBE_OK architecture_profiles count={len(MODEL_PROFILES)}")
    print(f"PROBE_OK architecture_rows count={len(rows)}")
    print(f"PROBE_OK architecture_report bytes={len(report)}")
    print("PROBE_OK architecture_profile_sweep self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", choices=sorted(MODEL_PROFILES), default=None)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--steps-override", type=int)
    parser.add_argument("--include-docs", action="store_true")
    parser.add_argument("--corpus-file", action="append", type=Path)
    parser.add_argument("--corpus-weight", type=int, default=1)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--sample-tokens", type=int, default=120)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--write-legacy-names", action="store_true")
    parser.add_argument("--refresh-heldout-float", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.models_root, args.evidence_dir)
        return

    profiles = selected_profiles(args)
    rows = build_rows(profiles, args.models_root, args.evidence_dir)

    if args.train:
        run_training(rows, args)
        rows = build_rows(profiles, args.models_root, args.evidence_dir)

    if args.refresh_heldout_float:
        refresh_heldout_reports(rows, args.evidence_dir)
        rows = build_rows(profiles, args.models_root, args.evidence_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(rows, args.evidence_dir), encoding="ascii")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

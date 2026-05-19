#!/usr/bin/env python3
"""Refresh all-suite host quality reports for exported GPT2-BASIC models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from audit_exported_models import DEFAULT_EVIDENCE, DEFAULT_MODELS_ROOT, find_root_models
from profile_pareto_report import model_key


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD = 0.72


@dataclass(frozen=True)
class RefreshResult:
    model: str
    output: Path
    status: int


def is_host_only_model(model_dir: Path) -> bool:
    marker = model_dir / "HOST_ONLY_NOT_DOS_READY.txt"
    profile = model_dir / "PROFILE.TXT"
    if marker.exists():
        return True
    if profile.exists() and "dos_ready=0" in profile.read_text(encoding="ascii", errors="ignore"):
        return True
    return False


def output_path_for_model(model_dir: Path, evidence_dir: Path) -> Path:
    if model_dir.name == "MODEL":
        return evidence_dir / "quality_report_default_model_all.md"
    return evidence_dir / f"quality_report_{model_key(model_dir)}_all.md"


def selected_models(models_root: Path, names: set[str], include_host_only: bool) -> list[Path]:
    models = []
    for model in find_root_models(models_root):
        if names and model.name not in names:
            continue
        if is_host_only_model(model.path) and not include_host_only:
            continue
        models.append(model.path)
    return models


def refresh_model(
    model_dir: Path,
    evidence_dir: Path,
    threshold: float,
    backend: str,
    device: str,
    dry_run: bool,
) -> RefreshResult:
    output = output_path_for_model(model_dir, evidence_dir)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_gpt2_basic_quality.py"),
        "--model-dir",
        str(model_dir),
        "--suite",
        "all",
        "--backend",
        backend,
        "--device",
        device,
        "--threshold",
        f"{threshold:.2f}",
        "--output",
        str(output),
    ]
    if dry_run:
        print(" ".join(command))
        return RefreshResult(model_dir.name, output, 0)

    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="ascii",
        errors="ignore",
        check=False,
    )
    print(f"QUALITY_REFRESH|model={model_dir.name}|status={process.returncode}|output={output.relative_to(ROOT)}")
    return RefreshResult(model_dir.name, output, process.returncode)


def self_test(models_root: Path, evidence_dir: Path) -> None:
    models = selected_models(models_root, set(), include_host_only=False)
    if not models:
        raise RuntimeError("no DOS-ready root models found")
    output = output_path_for_model(models[0], evidence_dir)
    if output.suffix != ".md":
        raise RuntimeError("quality output is not markdown")
    print(f"PROBE_OK refresh_quality_model_count={len(models)}")
    print(f"PROBE_OK refresh_quality_output={output.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--backend", choices=("float", "fixed"), default="float")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--include-host-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.models_root, args.evidence_dir)
        return

    models = selected_models(args.models_root, set(args.model), args.include_host_only)
    if args.limit > 0:
        models = models[: args.limit]
    if not models:
        raise SystemExit("QUALITY_REFRESH_FAILED no_models_selected")

    results = [
        refresh_model(model, args.evidence_dir, args.threshold, args.backend, args.device, args.dry_run)
        for model in models
    ]
    failures = [result for result in results if result.status != 0]
    print(f"QUALITY_REFRESH_SUMMARY|models={len(results)}|failures={len(failures)}")
    if failures:
        raise SystemExit("QUALITY_REFRESH_FAILED " + ",".join(result.model for result in failures))


if __name__ == "__main__":
    main()

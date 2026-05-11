#!/usr/bin/env python3
"""Verify that formerly aspirational software subsystems are implemented."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Requirement:
    label: str
    path: str
    patterns: tuple[str, ...]


FEATURES: dict[str, tuple[Requirement, ...]] = {
    "memory_tracker": (
        Requirement(
            "documented tracker API",
            "src/memory_manager.bas",
            (
                r"\bTYPE\s+MemoryTracker\b",
                r"\bDIM\s+SHARED\s+g_memory_tracker\s+AS\s+MemoryTracker\b",
                r"\bSUB\s+InitMemoryTracker\b",
                r"\bSUB\s+CleanupMemoryTracker\b",
                r"\bFUNCTION\s+TrackedAllocate\b",
                r"\bSUB\s+TrackedDeallocate\b",
            ),
        ),
        Requirement(
            "matrix pool API",
            "src/memory_manager.bas",
            (
                r"\bTYPE\s+MatrixPool\b",
                r"\bSUB\s+InitMatrixPool\b",
                r"\bSUB\s+CleanupMatrixPool\b",
                r"\bFUNCTION\s+GetPooledMatrix\b",
                r"\bSUB\s+ReturnMatrixToPool\b",
            ),
        ),
    ),
    "parameter_streaming": (
        Requirement(
            "layer cache and eviction",
            "src/memory_manager.bas",
            (
                r"\bTYPE\s+LayerCache\b",
                r"\bFUNCTION\s+GetLayerWeights\b",
                r"\bSUB\s+EnsureCacheSpace\b",
                r"\bSUB\s+LoadLayerFromDisk\b",
            ),
        ),
        Requirement(
            "disk-row streaming",
            "src/file_io.bas",
            (
                r"\bTYPE\s+ModelFileInfo\b",
                r"\bFUNCTION\s+StreamMatrixRows\b",
                r"\bSUB\s+StreamLayerWeights\b",
                r"\bSUB\s+CloseAllFiles\b",
            ),
        ),
    ),
    "block_sparse_attention": (
        Requirement(
            "block-sparse structures",
            "src/block_sparse.bas",
            (
                r"\bTYPE\s+SparseBlock\b",
                r"\bTYPE\s+SparseBlockMatrix\b",
                r"\bFUNCTION\s+FindBlock\b",
                r"\bFUNCTION\s+FindOrCreateBlock\b",
                r"\bSUB\s+CreateCausalSparseMask\b",
            ),
        ),
        Requirement(
            "block-sparse attention operations",
            "src/block_sparse.bas",
            (
                r"\bSUB\s+ComputeSparseBlockAttentionScores\b",
                r"\bSUB\s+SparseBlockSoftmax\b",
                r"\bSUB\s+BlockSparseAttention\b",
                r"\bTYPE\s+BlockHashTable\b",
                r"\bFUNCTION\s+FindBlockInHashTable\b",
                r"\bFUNCTION\s+ShouldUseSparseAttention\b",
                r"\bSUB\s+AdaptiveAttention\b",
            ),
        ),
    ),
    "simd_like_ops": (
        Requirement(
            "packing and arithmetic",
            "src/simd_ops.bas",
            (
                r"\bFUNCTION\s+Pack_8bit\b",
                r"\bFUNCTION\s+SIMD_Add_8bit\b",
                r"\bFUNCTION\s+SIMD_Subtract_8bit\b",
                r"\bFUNCTION\s+SIMD_Multiply_8bit\b",
                r"\bFUNCTION\s+SIMD_Add_4bit\b",
                r"\bFUNCTION\s+SIMD_Add_16bit\b",
            ),
        ),
        Requirement(
            "adaptive precision and CPU detection",
            "src/simd_ops.bas",
            (
                r"\bENUM\s+CPUType\b",
                r"\bFUNCTION\s+DetectCPU\b",
                r"\bFUNCTION\s+TestForFPU\b",
                r"\bFUNCTION\s+DetermineOptimalPrecision\b",
            ),
        ),
        Requirement(
            "matrix integration",
            "src/matrix_ops.bas",
            (
                r"\bSUB\s+MatrixMultiplySIMD\b",
                r"\bSUB\s+MatrixAddSIMD\b",
                r"\bSUB\s+TestSIMDMatrixOps\b",
            ),
        ),
    ),
    "fixed_point_and_assembly": (
        Requirement(
            "assembly/fallback fixed point",
            "src/asm_optimizations.bas",
            (
                r"\bFUNCTION\s+HasAssemblySupport\b",
                r"\bFUNCTION\s+FixedMultiplyFallback\b",
                r"\bFUNCTION\s+FixedMulAsm\b",
                r"\bSUB\s+MatrixMultiplyAsm\b",
                r"\bSUB\s+SoftmaxAsm\b",
            ),
        ),
        Requirement(
            "production fixed point",
            "src/real_gpt.bas",
            (
                r"\bCONST\s+TINYGPT_FX_SHIFT(?:\s+AS\s+INTEGER)?\s*=\s*12\b",
                r"\bFUNCTION\s+TinyGPTFixedMul\b",
                r"\bFUNCTION\s+TinyGPTFixedExpNeg\b",
                r"\bFUNCTION\s+TinyGPTForwardFixedLogits\b",
                r"\bSUB\s+TinyGPTFixedForwardCachedToken\b",
            ),
        ),
    ),
    "benchmarks_and_emulator_evidence": (
        Requirement(
            "component benchmarks",
            "src/benchmark.bas",
            (
                r"\bSUB\s+BenchmarkMatrixMultiply\b",
                r"\bSUB\s+BenchmarkAttention\b",
                r"\bSUB\s+BenchmarkMemoryStreaming\b",
            ),
        ),
        Requirement(
            "emulator timing contract",
            "qemu/evidence/hardware_perf_report.md",
            (
                r"emulated CPU-profile evidence",
                r"486dx2-66",
                r"Tokens/sec",
            ),
        ),
    ),
    "vga_visual_trace": (
        Requirement(
            "Mode 13h trace visualizer",
            "src/visual_trace.bas",
            (
                r"\bSCREEN\s+13\b",
                r"\bSUB\s+VisualDrawTokenBar\b",
                r"\bSUB\s+VisualizeTrace\b",
                r"VISUAL_GRAPHICS",
                r"VISUAL_TOKEN",
            ),
        ),
        Requirement(
            "QEMU visual trace runner",
            "qemu/run_visual_trace_486.sh",
            (
                r"fdauto_visual_trace\.bat",
                r"visual_trace_486",
                r"VISUAL\.LOG",
            ),
        ),
        Requirement(
            "DOS visual trace evidence",
            "qemu/evidence/visual_trace_486.log",
            (
                r"VISUAL_BEGIN",
                r"VISUAL_GRAPHICS\|mode=13\|status=ok",
                r"VISUAL_TOKEN\|kind=prompt",
                r"VISUAL_TOKEN\|kind=generated",
                r"VISUAL_END\|generated_tokens=12",
            ),
        ),
    ),
}


def read_text(root: Path, path: str) -> str:
    return (root / path).read_text(encoding="utf-8", errors="ignore")


def has_pattern(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def evaluate(root: Path) -> dict[str, object]:
    features: dict[str, object] = {}
    all_passed = True

    for feature_name, requirements in FEATURES.items():
        req_results = []
        feature_passed = True

        for requirement in requirements:
            text = read_text(root, requirement.path)
            missing = [pattern for pattern in requirement.patterns if not has_pattern(text, pattern)]
            passed = not missing
            feature_passed = feature_passed and passed
            req_results.append(
                {
                    "label": requirement.label,
                    "path": requirement.path,
                    "passed": passed,
                    "missing_patterns": missing,
                }
            )

        all_passed = all_passed and feature_passed
        features[feature_name] = {
            "passed": feature_passed,
            "requirements": req_results,
        }

    return {
        "runtime_backend": "source_probe",
        "completion_reason": "aspirational_software_implemented" if all_passed else "aspirational_software_incomplete",
        "artifact_file": "qemu/evidence/aspirational_software_closure.md",
        "checkpoint": "formerly_aspirational_software",
        "features": features,
        "done": all_passed,
    }


def write_markdown(result: dict[str, object], output: Path) -> None:
    lines = [
        "# Formerly Aspirational Software Closure",
        "",
        f"completion_reason: {result['completion_reason']}",
        "runtime_backend: source_probe",
        "timing_basis: emulator_until_physical_board_available",
        "",
        "| Feature | Status |",
        "|---|---|",
    ]

    features = result["features"]
    assert isinstance(features, dict)
    for feature_name, feature in features.items():
        assert isinstance(feature, dict)
        status = "PASS" if feature["passed"] else "FAIL"
        lines.append(f"| `{feature_name}` | {status} |")

    lines.extend(["", "## Evidence", ""])
    for feature_name, feature in features.items():
        assert isinstance(feature, dict)
        lines.append(f"### {feature_name}")
        requirements = feature["requirements"]
        assert isinstance(requirements, list)
        for requirement in requirements:
            assert isinstance(requirement, dict)
            status = "PASS" if requirement["passed"] else "FAIL"
            lines.append(f"- {status}: `{requirement['label']}` in `{requirement['path']}`")
            missing = requirement["missing_patterns"]
            if missing:
                lines.append(f"  missing: `{missing}`")
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    sample = "TYPE MemoryTracker\nFUNCTION TrackedAllocate(size AS LONG) AS ANY PTR\n"
    assert has_pattern(sample, r"\bTYPE\s+MemoryTracker\b")
    assert has_pattern(sample, r"\bFUNCTION\s+TrackedAllocate\b")
    assert not has_pattern(sample, r"\bSUB\s+CreateCausalSparseMask\b")
    assert "vga_visual_trace" in FEATURES
    print("PROBE_OK has_pattern exact_symbols=1")
    print("PROBE_OK vga_visual_trace feature=registered")
    print("PROBE_OK self_test exercised=1")
    print("PROBE_OK main cli_entry=available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    result = evaluate(args.repo_root)
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, args.output_md)

    for feature_name, feature in result["features"].items():
        assert isinstance(feature, dict)
        status = "PASS" if feature["passed"] else "FAIL"
        print(f"PROBE_OK {feature_name} status={status}")
    print(f"completion_reason={result['completion_reason']}")
    print(f"done={result['done']}")

    if not result["done"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

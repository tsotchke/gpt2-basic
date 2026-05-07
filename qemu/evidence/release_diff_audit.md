# Release Diff Audit

Date: 2026-05-06

Target: `gpt2-basic-486-production`

## Summary

The current working tree is a coherent release replacement, but it is still too
large for automatic guard-diff signoff. ICC completion/readiness evidence is
clean; the remaining production-audit blocker is intentional guard-diff policy:
the tracked diff contains 1,242 policy-deleted lines, above the 200-line release
limit.

This should not be suppressed in `.icc/production-audit.json`. The correct
resolution is to review and commit the release as a deliberate large replacement,
or split the source/runtime work into smaller commits with evidence preserved.

## Current Diff Shape

Tracked source/doc/config diff:

| Scope | Files | Insertions | Deletions |
|---|---:|---:|---:|
| tracked changes | 16 | 2,638 | 1,331 |

Largest policy-deletion contributors:

| Path | Policy-deleted lines |
|---|---:|
| `src/benchmark.bas` | 236 |
| `src/model.bas` | 229 |
| `src/file_io.bas` | 165 |
| `src/main.bas` | 133 |
| `src/transformer_components.bas` | 103 |
| `README.md` | 89 |
| `src/tokenizer.bas` | 84 |
| `src/memory_manager.bas` | 58 |
| `src/simd_ops.bas` | 32 |
| `src/softmax_fixed.bas` | 31 |

Untracked release payload:

| Classification | Files |
|---|---:|
| candidate model snapshots | 91 |
| curriculum files | 8 |
| generated local bytecode | 26 |
| host tooling | 20 |
| production source | 3 |
| promoted model files | 3 |
| QEMU tooling | 21 |
| ICC repo config | 2 |
| runtime evidence | 110 |

ICC release classification now covers all changed paths: unknown changed paths
are `0`.

## Source Surface Audit

Symbol-level deltas show replacement work rather than silent removal of major
runtime capability.

| File | Added symbols | Removed symbols | Notes |
|---|---|---|---|
| `src/asm_optimizations.bas` | `FixedAdd`, `FixedDivide`, `FixedMultiply`, `FixedSubtract` | `IIF` | Replaces non-portable helper with explicit fixed-point helpers and DOS-compatible operators. |
| `src/benchmark.bas` | `BenchmarkGPT2BasicRuntime`, `BenchmarkGenerateText`, `CleanupBenchmarkFiles`, `CreateBenchmarkLayerFiles` | `CleanupTestFiles`, `CreateTestLayerFiles`, `GenerateText` | Replaces synthetic transformer block benchmark path with production GPT2-BASIC runtime benchmark path. |
| `src/data_structures.bas` | `CEILING`, `CompatFormat`, `IIFString`, `MAX`, `MIN`, `TANH` | `IIF` | Adds compiler-compatible utility surface used across DOS runtime code. |
| `src/file_io.bas` | `CreateDiagnosticModel`, `InitGPT2BasicModelFiles`, `LoadModelVocabulary`, `ModelDirectoryHasGPT2BasicCheckpoint` | `CreateDummyModel`, `LoadVocabulary` | Replaces dummy/test naming with diagnostic path and adds GPT2-BASIC checkpoint discovery/config loading. |
| `src/main.bas` | `PerfDoubleText`, `PerfLongText`, `PrintActiveModelInfo`, `RunAutomatedQualitySuite`, `RunHardwarePerformanceCase`, `RunHardwarePerformanceSuite`, `RunQualityPrompt`, `StripTrailingEOT` | none | Adds real runtime quality and hardware performance surfaces. |
| `src/model.bas` | `GenerateModelText`, `LoadTextModelConfig`, `SoftmaxVectorFixedPoint` | `GenerateText`, `InitModelConfig`, `LoadModelConfig` | Renames generation/config entry points around GPT2-BASIC text config and fixed-point logits. |
| `src/quantization.bas` | `DequantizeLogToFixed` | none | Adds fixed-point bridge needed by runtime components. |
| `src/tokenizer.bas` | `CleanTokenizerText`, `LexiconTokenize`, `TokenPieceBoundaryMatches`, `TokenPieceMatchesBytes`, `TokenizerLexiconWordByte`, `TokenizerOutputAllowed` | none | Adds byte cleaning, lexicon mode, BPE/lexicon output legality, and DOS tokenizer parity hooks. |
| `src/transformer_components.bas` | `ShouldUseBlockSparseAttention` | none | Adds runtime decision point for block-sparse attention. |

Files with no symbol removals: `src/block_sparse.bas`, `src/matrix_ops.bas`,
`src/memory_manager.bas`, `src/simd_ops.bas`, `src/softmax_fixed.bas`.

## ICC Status

Latest ICC status after `.icc` policy scoping:

| Gate | Result |
|---|---|
| completion oracle | complete, 5/5 pass |
| readiness | ready, score 100 |
| source drift | clean |
| release classification | pass, unknown 0 |
| index quality | pass, blind spots 0 |
| audit patterns | pass for shell hardening, Python stubs, and Python leakage |
| production audit | fail only on guard-diff deletion volume |

## Release Decision

The current diff is structurally defensible as a release replacement, but it is
not eligible for automatic guard-diff clearance. Before final release signoff,
do one of the following:

1. Split into reviewable commits:
   - DOS compatibility/core helpers.
   - GPT2-BASIC fixed-point runtime and file format support.
   - Tokenizer modes and output mask contract.
   - Host training/evaluation/export tooling.
   - Model artifacts, QEMU evidence, and `.icc` release policy.
2. Or explicitly accept a single large replacement commit with this audit,
   compile/vector/quality evidence, and ICC readiness attached.

Do not suppress `guard_diff`; it is correctly reporting the size of the
uncommitted release change.

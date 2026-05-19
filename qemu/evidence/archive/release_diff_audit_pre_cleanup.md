# Production Audit: `gpt2_basic`

Historical note: this is a pre-cleanup ICC report retained for provenance. It
does not describe the current workspace state after the DOS preview cleanup
commits. The current tracking summary is
`qemu/evidence/workspace_tracking_audit.md`.

Verdict: `fail`  Target: `gpt2-basic-486-production`  Risks: `5`
Policy: `gpt2-basic-486-production`  Source: `<repo>/.icc/production-audit.json`  Suppressed: `0`

| Severity | Area | Finding | Action |
|---|---|---|---|
| `high` | `source_drift` | 3 indexed file(s) changed since artifact refresh | review changed paths and refresh artifacts after the edits are intentional |
| `high` | `source_drift` | indexed artifacts are stale relative to the live source tree | inspect source-drift, then refresh index, memory, architecture, and git-history artifacts |
| `medium` | `forward_progress` | 22 rollback-sensitive diff risk(s); 0 block production signoff | replace rollback-shaped changes with complete optimal implementation, or prove the architectural replacement in the same diff |
| `medium` | `guard_diff` | 3 guard warning(s) in the current git diff | review warning details before merging |
| `low` | `guard_diff` | 217 file(s) differ from HEAD | commit or discard intentional local edits before final production signoff |

## Evidence Summary

| Surface | Status | Key Counts |
|---|---|---|
| `artifacts` | `PASS` | stale `True` |
| `source_drift` | `STALE` | changed `3` added `0` modified `3` deleted `0` |
| `guard_diff` | `PASS` | violations `0` warnings `3` |
| `index_quality` | `PASS` | blind spots `0` |
| `release_classification` | `PASS` | changed `272` unknown `0` |
| `feature_map` | `PASS` | clusters `12` promotion candidates `0` |
| `tasks` | `portfolio` | tasks `0` returned `0` |
| `readiness` | `ready` | score `100` gaps `0` stubs `0` runtime checks `2` |
| `audit_patterns:shell-hardening` | `PASS` | findings `0` severities `{}` |
| `audit_patterns:python-stub-bodies` | `PASS` | findings `0` severities `{}` |
| `audit_patterns:python-production-leakage` | `PASS` | findings `0` severities `{}` |

## Release Classification

Rules: `17`  Require classified changes: `True`

| Role | Changed Paths |
|---|---:|
| `assistant_pack` | 23 |
| `candidate_model` | 29 |
| `curriculum` | 2 |
| `documentation` | 11 |
| `full_release_model` | 1 |
| `hardware_capture` | 3 |
| `host_tests` | 4 |
| `host_tooling` | 16 |
| `production_source` | 6 |
| `promoted_model` | 4 |
| `qemu_tooling` | 7 |
| `repo_config` | 3 |
| `runtime_evidence` | 155 |
| `speed_release_model` | 8 |

## Guard Diff Detail

| Area | Files | Added | Modified | Deleted | Untracked | +/- Lines | Policy Del | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `qemu` | 162 | 0 | 23 | 0 | 139 | +8407 | 272 | `qemu/README.md`, `qemu/evidence/architecture_codebase_audit.md`, `qemu/evidence/architecture_profile_sweep_probe.log`, `qemu/evidence/compile_main_486.log` |
| `assets` | 16 | 0 | 7 | 0 | 9 | +52 | 69 | `assets/gpt2_basic/MODEL/GPT2FX.BIN`, `assets/gpt2_basic/MODEL/GPT2VEC.TXT`, `assets/gpt2_basic/MODEL/GPT2WT.BIN`, `assets/gpt2_basic/MODEL/VOCAB.BIN` |
| `scripts` | 16 | 0 | 5 | 0 | 11 | +3351 | 22 | `scripts/architecture_profile_sweep.py`, `scripts/evaluate_gpt2_basic_quality.py`, `scripts/export_gpt2_basic_vectors.py`, `scripts/model_report.py` |
| `repo-root` | 9 | 0 | 9 | 0 | 0 | +234 | 28 | `README.md`, `gpt2_basic_documentation.md`, `gpt2_basic_tldr.md`, `implementation_guide.md` |
| `src` | 6 | 0 | 4 | 0 | 2 | +1372 | 9 | `src/asm_optimizations.bas`, `src/block_sparse.bas`, `src/memory_manager.bas`, `src/real_gpt.bas` |
| `data` | 2 | 0 | 0 | 0 | 2 | +828 | 0 | `data/domain_curriculum/cache_repair_focus_v1.txt`, `data/domain_curriculum/gold_curriculum_v5_clean_repair.txt` |
| `repo-oracle-config` | 2 | 0 | 2 | 0 | 0 | +96 | 2 | `.icc/completion-oracles.json`, `.icc/production-audit.json` |
| `.github` | 1 | 0 | 0 | 0 | 1 | +0 | 0 | `.github/` |
| `codebase-tool-tests` | 1 | 0 | 0 | 0 | 1 | +0 | 0 | `tests/` |
| `docs` | 1 | 0 | 0 | 0 | 1 | +0 | 0 | `docs/` |

| Path | Raw Deleted | Moved Discount | Policy Deleted |
|---|---:|---:|---:|
| `qemu/evidence/release_diff_audit.md` | 109 | 4 | 105 |
| `assets/gpt2_basic/MODEL/GPT2VEC.TXT` | 42 | 0 | 42 |
| `qemu/evidence/quality_486.log` | 129 | 95 | 34 |
| `qemu/evidence/architecture_codebase_audit.md` | 26 | 0 | 26 |
| `qemu/evidence/profile_pareto_report.md` | 17 | 0 | 17 |
| `README.md` | 15 | 0 | 15 |
| `assets/gpt2_basic/MODEL_SUBWORD512_PROTO/quality_heldout.md` | 14 | 0 | 14 |
| `assets/gpt2_basic/MODEL_SUBWORD512_PROTO/quality_runtime.md` | 14 | 1 | 13 |
| `qemu/README.md` | 13 | 0 | 13 |
| `qemu/evidence/icc_readiness.md` | 22 | 9 | 13 |

## Feature Map Detail

| Surface | Role | Files | Lines | Promotions |
|---|---|---:|---:|---:|
| `scripts/train_tiny_gpt.py` | `tooling_surface` | 1 | 1076 | 0 |
| `scripts/evaluate_gpt2_basic_quality.py` | `tooling_surface` | 1 | 684 | 0 |
| `scripts/train_subword_prototype.py` | `tooling_surface` | 1 | 486 | 0 |
| `scripts/build_preview_release.py` | `tooling_surface` | 1 | 416 | 0 |
| `scripts/train_assistant_pack_models.py` | `tooling_surface` | 1 | 702 | 0 |
| `scripts/architecture_profile_sweep.py` | `tooling_surface` | 1 | 449 | 0 |
| `scripts/plan_model_quality_repairs.py` | `tooling_surface` | 1 | 212 | 0 |
| `src/real_gpt.bas` | `product_surface` | 1 | 3485 | 0 |
| `scripts/model_report.py` | `tooling_surface` | 1 | 536 | 0 |
| `scripts/audit_exported_models.py` | `tooling_surface` | 1 | 358 | 0 |

| Cluster | Role | Files | Lines | Prefixes | Promotions |
|---:|---|---:|---:|---|---:|
| 10 | `docs_surface` | 546 | 95467 | `qemu`, `assets`, `data`, `memory-bank` | 0 |
| 4 | `product_surface` | 18 | 17866 | `src` | 0 |
| 8 | `tooling_surface` | 33 | 12755 | `scripts`, `qemu` | 0 |
| 2 | `docs_surface` | 2 | 3867 | `README.md`, `gpt2_basic_documentation.md` | 0 |
| 7 | `docs_surface` | 3 | 1844 | `implementation_guide.md`, `memory_management_design.md`, `src` | 0 |
| 6 | `mixed_surface` | 12 | 1038 | `qemu`, `to_latex.sh` | 0 |
| 11 | `docs_surface` | 1 | 1022 | `qemu` | 0 |
| 9 | `product_surface` | 1 | 931 | `src` | 0 |
| 0 | `tooling_surface` | 1 | 684 | `scripts` | 0 |
| 5 | `mixed_surface` | 2 | 411 | `assets`, `data` | 0 |

## Recommended Commands
- `python3 scripts/codebase_tool.py index --repo gpt2_basic --base-sha 75dd0969ba4e24a51788b0866e5b13dd8996ac07`
- `python3 scripts/codebase_tool.py build-memory --repo gpt2_basic`
- `python3 scripts/codebase_tool.py build-git-history --repo gpt2_basic`

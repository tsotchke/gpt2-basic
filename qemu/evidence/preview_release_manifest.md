# GPT2-BASIC Preview Release Manifest

Version: `v0.1.0-preview`
Generated: `2026-05-12`
Package tree: `gpt2-basic-preview`
Package zip: `gpt2-basic-preview.zip`
Package checksums: `SHA256SUMS.txt`; zip sidecar: `gpt2-basic-preview.zip.sha256`
Package status: `296 files, 81,132,595 bytes`

This is an iterative preview payload. It ships only strict-quality release models and assistant packs; rejected repair attempts and old candidates remain repo evidence only.

## Release Models

| Model | Label | Shape | Quality | Size | Why shipped |
|---|---|---|---|---|---|
| `MODEL` | default full resident | `2L 48D 4H ctx192 h192 v4096` | PASS 10/10 avg 0.969 (dos-fixed-qemu, all) | 7 files / 3,814,630 B | current slim production default |
| `MODEL_LEXICON_GOLD_V4_S3000` | gold-v4 full resident | `2L 48D 4H ctx192 h192 v4096` | PASS 10/10 avg 0.969 (dos-fixed-qemu, all) | 7 files / 3,814,630 B | best measured full-resident quality checkpoint |
| `MODEL_HEADSHORTLIST2048_PROD_PROBE` | head shortlist | `2L 48D 4H ctx192 h192 v4096` | PASS 10/10 avg 0.960 (float, all) | 8 files / 3,823,256 B | fastest measured large-vocabulary path |
| `MODEL_HEADQ4_PROD_PROBE` | q4 output head | `2L 48D 4H ctx192 h192 v4096` | PASS 10/10 avg 0.960 (float, all) | 8 files / 3,929,420 B | compressed output-head candidate |
| `MODEL_TOKHEADQ4_PROD_PROBE` | q4 token+head | `2L 48D 4H ctx192 h192 v4096` | PASS 10/10 avg 0.960 (float, all) | 9 files / 4,044,296 B | best current low-memory default |
| `MODEL_TOKHEADQ4_STREAM_PROD_PROBE` | q4 streamed head | `2L 48D 4H ctx192 h192 v4096` | PASS 10/10 avg 0.960 (float, all) | 10 files / 4,044,376 B | maximum RAM compatibility fallback |

## Assistant Packs

| Pack Model | Shape | Quality | Size |
|---|---|---|---|
| `ASSISTANT_CHAT` | `3L 64D 4H ctx192 h256 v4096` | NEEDS_TRAINING 56/72 avg 0.879 (float, assistant-pack) | 16 files / 45,564,461 B |
| `ASSISTANT_DOSHELP` | `2L 48D 4H ctx192 h192 v4096` | PASS 4/4 avg 1.000 (float, assistant-pack) | 14 files / 3,994,189 B |
| `ASSISTANT_OFFICE` | `2L 48D 4H ctx192 h192 v4096` | PASS 4/4 avg 1.000 (float, assistant-pack) | 14 files / 4,141,696 B |

## Included Runtime Surface

- `bin/GPT2.EXE` when current QEMU evidence includes the compiled DOS binary.
- `assets/gpt2_basic/MODEL*` for the release models listed above.
- `assets/gpt2_basic/PACKS` with CHAT, DOSHELP, and OFFICE packs, per-pack `USAGE.TXT`, pack-local models where available, and sprite/icon slots.
- `src`, `scripts`, `tests`, selected `qemu` helpers, and `data/domain_curriculum` for rebuild and repair iteration.
- `docs/dosbox.md` and `scripts/build_dosbox_bundle.py` for the DOSBox convenience package.
- Selected QEMU and quality evidence under `qemu/evidence`.

## Deferred Release Scope

- This preview release is the DOS demo, DOSBox convenience package, and DOS transfer package.
- Windows and OS/2 assistant shells, including the OS/2/Warp package, stay on the later-release track.

## Explicitly Excluded

- Historical byte/domain candidates that miss the strict all-suite quality gate.
- Rejected repair attempts such as `MODEL_PROFILE_386_MIN_LEXICON384_REPAIR`.
- Host-only prototypes that are not DOS-ready.

## Physical Hardware Path

- `docs/hardware-validation.md` defines the non-emulator validation ladder.
- A physical 486-class DOS machine is the solid-release baseline.
- `qemu/run_hardware_capture_486.sh` rehearses the same `C:\GPT2\HWVALID.BAT` capture path before physical transfer.
- `scripts/build_hardware_transfer.py` creates the minimal `C:\GPT2` transfer bundle for the physical machine.
- `C:\GPT2\RETURN.TXT` gives the DOS-side copy-back checklist for real machine logs.
- `scripts/stage_hardware_capture_evidence.py` verifies returned physical logs and stages stable release evidence names.
- `scripts/hardware_performance_matrix.py` builds the physical-only performance table from staged logs.
- Pentium hardware is useful for scaling evidence, but it is not a blocker for the 486-focused release.

## Release Notes

- `docs/releases/v0.1.0-preview.md` is the GitHub release body for this payload.
- Attach the preview zip, DOSBox zip, hardware-transfer zip, `.sha256` sidecars, and `preview_release_manifest.md` to the GitHub prerelease.

## Evidence Files

- `qemu/evidence/GPT2.EXE`
- `qemu/evidence/assistant_486.log`
- `qemu/evidence/assistant_chat_manual_probe_2026-05-12.md`
- `qemu/evidence/assistant_compile_486.log`
- `qemu/evidence/assistant_interactive_chat_486.md`
- `qemu/evidence/assistant_pack_probe.log`
- `qemu/evidence/assistant_raw_prompt_eval.md`
- `qemu/evidence/assistant_stress_486.log`
- `qemu/evidence/assistant_stress_compile_486.log`
- `qemu/evidence/assistant_stress_report.md`
- `qemu/evidence/compile_main_486.log`
- `qemu/evidence/exported_model_quality_inventory.md`
- `qemu/evidence/gold_curriculum_v5_clean_repair_report.md`
- `qemu/evidence/hardware_capture_486_qemu/ASSIST.LOG`
- `qemu/evidence/hardware_capture_486_qemu/ASSISTC.LOG`
- `qemu/evidence/hardware_capture_486_qemu/HWNOTES.TXT`
- `qemu/evidence/hardware_capture_486_qemu/HWVALID.LOG`
- `qemu/evidence/hardware_capture_486_qemu/PERF.LOG`
- `qemu/evidence/hardware_capture_486_qemu/QUAL.LOG`
- `qemu/evidence/hardware_capture_486_qemu_probe.log`
- `qemu/evidence/hardware_capture_probe.log`
- `qemu/evidence/hardware_perf_report.md`
- `qemu/evidence/hardware_performance_matrix.md`
- `qemu/evidence/hardware_transfer_probe.log`
- `qemu/evidence/improvement_backlog.md`
- `qemu/evidence/model_quality_repair_plan.md`
- `qemu/evidence/perf_486_486dx2-66_model_lexicon_gold_v4_s3000.log`
- `qemu/evidence/preview_release_probe.log`
- `qemu/evidence/quality_486_model_lexicon_gold_v4_s3000.log`
- `qemu/evidence/quality_report_assistant_chat.md`
- `qemu/evidence/quality_report_assistant_doshelp.md`
- `qemu/evidence/quality_report_assistant_office.md`
- `qemu/evidence/quality_report_default_model_all.md`
- `qemu/evidence/quality_report_default_model_fixed_all.md`
- `qemu/evidence/quality_report_dos.md`
- `qemu/evidence/quality_report_dos_all.md`
- `qemu/evidence/quality_report_dos_heldout.md`
- `qemu/evidence/quality_report_dos_model_lexicon_gold_v4_s3000.md`
- `qemu/evidence/quality_report_dos_model_lexicon_gold_v4_s3000_all.md`
- `qemu/evidence/quality_report_dos_model_lexicon_gold_v4_s3000_heldout.md`
- `qemu/evidence/quality_report_headq4_prod_probe_all.md`
- `qemu/evidence/quality_report_headq4_prod_probe_fixed_all.md`
- `qemu/evidence/quality_report_headq4_prod_probe_heldout.md`
- `qemu/evidence/quality_report_headq4_prod_probe_runtime.md`
- `qemu/evidence/quality_report_headshortlist2048_heldout.md`
- `qemu/evidence/quality_report_headshortlist2048_prod_probe_all.md`
- `qemu/evidence/quality_report_headshortlist2048_runtime.md`
- `qemu/evidence/quality_report_lexicon_gold_v4_s3000_all.md`
- `qemu/evidence/quality_report_lexicon_gold_v4_s3000_fixed_all.md`
- `qemu/evidence/quality_report_lexicon_gold_v4_s3000_heldout.md`
- `qemu/evidence/quality_report_lexicon_gold_v4_s3000_runtime.md`
- `qemu/evidence/quality_report_tokheadq4_prod_probe_all.md`
- `qemu/evidence/quality_report_tokheadq4_prod_probe_fixed_all.md`
- `qemu/evidence/quality_report_tokheadq4_prod_probe_heldout.md`
- `qemu/evidence/quality_report_tokheadq4_prod_probe_runtime.md`
- `qemu/evidence/quality_report_tokheadq4_stream_prod_probe_all.md`
- `qemu/evidence/run_main_486.log`
- `qemu/evidence/vector_486_model_lexicon_gold_v4_s3000.log`
- `qemu/evidence/workspace_tracking_probe.log`

## Verification Commands

```sh
python3 -m unittest discover tests
python3 scripts/audit_exported_models.py
python3 scripts/verify_assistant_packs.py
python3 scripts/verify_hardware_capture.py --self-test
python3 scripts/stage_hardware_capture_evidence.py --self-test
python3 scripts/hardware_performance_matrix.py --self-test
python3 scripts/build_dosbox_bundle.py --self-test
python3 scripts/build_hardware_transfer.py --self-test
python3 scripts/build_preview_release.py --self-test
python3 scripts/verify_preview_artifacts.py --self-test
python3 scripts/verify_workspace_tracking.py
python3 scripts/build_preview_release.py --force
python3 scripts/build_dosbox_bundle.py --force
python3 scripts/build_hardware_transfer.py --force
python3 scripts/verify_preview_artifacts.py
```

Audited models: `47`

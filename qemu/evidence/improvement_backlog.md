# GPT2-BASIC Improvement Backlog

This backlog is evidence-driven. The preview release can iterate now, while low-quality exports stay in the repair or retirement lanes instead of being presented as release options.

## Current State

- Models audited: `46`
- Strict quality pass: `11/46`
- Release-ready root models: `6`
- Assistant pack models passing: `2`
- Priority retrain/experiment queue: `7`
- Rejected repair attempts recorded as negative evidence: `6`
- Retired or host-only exports: `23`

## Iteration Lanes

| Lane | Immediate goal | Evidence gate |
|---|---|---|
| Preview release | Package only passing release models, assistant packs, source, and selected QEMU evidence. | `python3 scripts/build_preview_release.py --manifest-only` and `--force` for a package tree. |
| Model quality | Retrain 386-min and 486DX2 profiles from the gold curriculum; keep rejected repairs out of release payloads. | `python3 scripts/refresh_model_quality_reports.py` then `python3 scripts/plan_model_quality_repairs.py`. |
| Runtime speed/RAM | Keep measuring full, head-shortlist, q4 token+head, and q4 streamed modes on 486 profiles. | `bash qemu/run_perf_486.sh 486dx2-66 <model-dir>` and `qemu/evidence/hardware_perf_report.md`. |
| Assistant packs | Improve DOSHELP/OFFICE prompt corpora, action routing, and sprite/icon renderer without bloating `GPT2.EXE`. | `bash qemu/run_assistant_486.sh` and `python3 scripts/verify_assistant_packs.py`. |
| Windows/OS2 shells | Reuse `PACK.INI`, `HELP.TXT`, `SPRITE`, `ICONS`, and pack-local model directories in native shells. | Shared pack parser tests plus one DOS, one Windows, and one OS/2 scripted probe. |
| Real hardware | Move the emulator-passing release to one physical 486-class DOS machine first; Pentium data is optional scaling evidence. | `docs/hardware-validation.md` plus real-machine `--quality-all`, `--perf`, and assistant logs. |

## Release Payload

- `MODEL`: PASS 10/10 avg 0.969 (dos-fixed-qemu, all)
- `MODEL_HEADQ4_PROD_PROBE`: PASS 10/10 avg 0.960 (float, all)
- `MODEL_HEADSHORTLIST2048_PROD_PROBE`: PASS 10/10 avg 0.960 (float, all)
- `MODEL_LEXICON_GOLD_V4_S3000`: PASS 10/10 avg 0.969 (dos-fixed-qemu, all)
- `MODEL_TOKHEADQ4_PROD_PROBE`: PASS 10/10 avg 0.960 (float, all)
- `MODEL_TOKHEADQ4_STREAM_PROD_PROBE`: PASS 10/10 avg 0.960 (float, all)

## Candidate Promotion Queue

- `MODEL_LEXICON_GOLD_V1_S3000`: PASS 10/10 avg 0.890 (float, all); needs DOS vector/perf evidence before release promotion.
- `MODEL_LEXICON_GOLD_V3_S3000`: PASS 10/10 avg 0.930 (float, all); needs DOS vector/perf evidence before release promotion.

## Retrain Queue

- `MODEL_BPE384_COMPLETE_W2S3000`: NEEDS_TRAINING 0/10 avg 0.739 (float, all); `retrain_experiment`; BPE path is an experiment; retrain only if comparing tokenizer families is still useful.
- `MODEL_BPE384_DOMAIN_W2S3000`: NEEDS_TRAINING 8/10 avg 0.794 (float, all); `retrain_experiment`; BPE path is an experiment; retrain only if comparing tokenizer families is still useful.
- `MODEL_LEXICON384_W2S3000`: NEEDS_TRAINING 2/10 avg 0.769 (float, all); `retrain_optional`; misses strict all-suite but has no release-critical role.
- `MODEL_LEXICON4096_BOUNDARY_W32S3000`: NEEDS_TRAINING 1/10 avg 0.810 (float, all); `retrain_optional`; misses strict all-suite but has no release-critical role.
- `MODEL_LEXICON4096_W2S3000`: NEEDS_TRAINING 6/10 avg 0.829 (float, all); `retrain_optional`; misses strict all-suite but has no release-critical role.
- `MODEL_PROFILE_386_MIN`: NEEDS_TRAINING 1/10 avg 0.612 (dos-fixed-qemu, all); `retrain_priority`; fastest measured profile is valuable only if a small-vocabulary repair passes quality.
- `MODEL_PROFILE_486DX2_USABLE`: NEEDS_TRAINING 2/10 avg 0.715 (dos-fixed-qemu, all); `retrain_priority`; larger 486DX2 profile should be retrained from the clean gold curriculum, not old byte data.

## Rejected Repairs To Keep Out Of Release

- `MODEL_LEXICON4096_ADAPTED_REPAIR2_S600`: NEEDS_TRAINING 1/10 avg 0.768 (float, all)
- `MODEL_LEXICON4096_ADAPTED_REPAIR_S900`: NEEDS_TRAINING 7/10 avg 0.844 (float, all)
- `MODEL_PROFILE_386_MIN_LEXICON384_REPAIR`: NEEDS_TRAINING 2/10 avg 0.714 (float, all)
- `MODEL_PROFILE_486DX2_CACHEFOCUS_REPAIR`: NEEDS_TRAINING 6/10 avg 0.805 (float, all)
- `MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR`: NEEDS_TRAINING 9/10 avg 0.916 (float, all)
- `MODEL_PROFILE_486DX2_LEXICON4096_REPAIR`: NEEDS_TRAINING 9/10 avg 0.894 (float, all)

## Decision Counts

- `rejected_repair`: 6
- `keep`: 2
- `keep_candidate`: 2
- `keep_release`: 6
- `exclude_host_only`: 1
- `retire_superseded`: 22
- `retrain_experiment`: 2
- `retrain_optional`: 3
- `retrain_priority`: 2

## Next Commands

```sh
python3 scripts/build_quality_repair_corpus.py
python3 scripts/build_preview_release.py --manifest-only
python3 scripts/write_improvement_backlog.py
python3 scripts/train_tiny_gpt.py --profile 486dx2-usable --tokenizer lexicon --vocab-size 4096 --lexicon-min-count 1 --base-weight 0 --corpus-file data/domain_curriculum/gold_curriculum_v5_clean_repair.txt --corpus-weight 1 --steps 7000 --log-every 500 --output assets/gpt2_basic/MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR2
python3 scripts/evaluate_gpt2_basic_quality.py --model-dir assets/gpt2_basic/MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR2 --suite all --backend float --threshold 0.72 --output qemu/evidence/quality_report_profile_486dx2_cleanlexicon4096_repair2_all.md
python3 scripts/plan_model_quality_repairs.py
bash qemu/run_assistant_486.sh
```

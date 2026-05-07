# GPT2-BASIC Profile Pareto Report

This report ranks exported checkpoints using shape-derived memory/work estimates, available QEMU `--perf` measurements, and held-out quality evidence.

QEMU speed is per-model when a profile-specific `--perf` log exists; otherwise the row uses the shape-derived model estimate. It is still emulator evidence, not physical-board timing.

## Ranking

| Rank | Model dir | Shape | Params | Runtime bytes | 486DX2 tok/s | Host tok/s | Regression quality | Held-out quality | Score |
|---:|---|---|---:|---:|---:|---:|---|---|---:|
| 1 | `MODEL_PROFILE_386_MIN` | `2L 32D 4H ctx128 h128 v258` | 46338 | 259108 | 11.70 measured | missing | NEEDS_TRAINING 1/5 avg 0.645 (dos-fixed-qemu) | NEEDS_TRAINING 0/5 avg 0.579 (dos-fixed-qemu) | 27.42 |
| 2 | `MODEL_DOMAIN_FAMILY_W2S1000` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | NEEDS_TRAINING 3/5 avg 0.763 (dos-fixed-qemu) | NEEDS_TRAINING 4/5 avg 0.850 (dos-fixed-qemu) | 9.28 |
| 3 | `MODEL_BASELINE_PRE_FINETUNE1` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 3/5 avg 0.797 (float) | 8.70 |
| 4 | `MODEL_CANDIDATE_MPS` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | NEEDS_TRAINING 4/5 avg 0.840 (float) | NEEDS_TRAINING 3/5 avg 0.797 (float) | 8.70 |
| 5 | `MODEL_CANDIDATE_FINETUNE1` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | PASS 5/5 avg 0.889 (float) | NEEDS_TRAINING 2/5 avg 0.763 (float) | 8.33 |
| 6 | `MODEL_DOMAIN_ANCHOR_W2S1000` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 2/5 avg 0.752 (float) | 8.21 |
| 7 | `MODEL_DOMAIN_RETENTION_S250` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 2/5 avg 0.745 (float) | 8.14 |
| 8 | `MODEL_DOMAIN_CANDIDATE` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 2/5 avg 0.741 (float) | 8.09 |
| 9 | `MODEL_CANDIDATE_MPS3` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | NEEDS_TRAINING 3/5 avg 0.813 (float) | NEEDS_TRAINING 1/5 avg 0.716 (float) | 7.82 |
| 10 | `MODEL_DOMAIN_ANSWERBANK_W2S1000` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 1/5 avg 0.711 (float) | 7.76 |
| 11 | `MODEL_BPE384_COMPLETE_W2S3000` | `2L 48D 4H ctx192 h192 v384` | 103104 | 571140 | 5.19 model | missing | missing | NEEDS_TRAINING 3/5 avg 0.812 (float) | 7.73 |
| 12 | `MODEL_DOMAIN_W2S800` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 1/5 avg 0.708 (float) | 7.73 |
| 13 | `MODEL_CANDIDATE` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | NEEDS_TRAINING 2/5 avg 0.744 (float) | NEEDS_TRAINING 1/5 avg 0.691 (float) | 7.55 |
| 14 | `MODEL_BASELINE_PRE_MPS` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 0/5 avg 0.681 (float) | 7.44 |
| 15 | `MODEL_CANDIDATE_MPS2` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | NEEDS_TRAINING 4/5 avg 0.838 (float) | NEEDS_TRAINING 0/5 avg 0.676 (float) | 7.38 |
| 16 | `MODEL_DOMAIN_FAMILY2_W2S1000` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 0/5 avg 0.667 (float) | 7.28 |
| 17 | `MODEL_DOMAIN_REHEARSAL` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 0/5 avg 0.655 (float) | 7.15 |
| 18 | `MODEL_BPE384_DOMAIN_W2S3000` | `2L 48D 4H ctx192 h192 v384` | 103104 | 571140 | 5.19 model | missing | missing | NEEDS_TRAINING 1/5 avg 0.748 (float) | 7.12 |
| 19 | `MODEL_ONLINE_CANDIDATE` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | NEEDS_TRAINING 0/5 avg 0.650 (float) | 7.10 |
| 20 | `MODEL_LEXICON384_W2S3000` | `2L 48D 4H ctx192 h192 v384` | 103104 | 571140 | 5.19 model | missing | missing | NEEDS_TRAINING 1/5 avg 0.726 (float) | 6.91 |
| 21 | `MODEL_LEXICON_GOLD_V1_S3000` | `2L 48D 4H ctx192 h192 v2710` | 328726 | 1501540 | 2.87 model | missing | missing | NEEDS_TRAINING 2/5 avg 0.923 (float) | 1.85 |
| 22 | `MODEL_PROFILE_486DX2_USABLE` | `3L 64D 4H ctx192 h256 v258` | 195650 | 1088292 | 2.65 measured | missing | NEEDS_TRAINING 2/5 avg 0.757 (dos-fixed-qemu) | NEEDS_TRAINING 0/5 avg 0.673 (dos-fixed-qemu) | 1.72 |
| 23 | `MODEL_486DX2_DOMAIN_W2S1200` | `3L 64D 4H ctx192 h256 v258` | 195650 | 1088292 | 2.37 model | missing | missing | NEEDS_TRAINING 0/5 avg 0.700 (float) | 1.60 |
| 24 | `MODEL` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.48 measured | missing | PASS 5/5 avg 0.948 (dos-fixed-qemu) | PASS 5/5 avg 0.973 (dos-fixed-qemu) | 1.23 |
| 25 | `MODEL_LEXICON4096_BOUNDARY_W32S3000` | `2L 48D 4H ctx192 h192 v3768` | 431352 | 1924740 | 2.39 model | missing | missing | NEEDS_TRAINING 1/5 avg 0.875 (float) | 1.14 |
| 26 | `MODEL_LEXICON_GOLD_V2_S3000` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | PASS 5/5 avg 0.948 (dos-fixed-qemu) | PASS 5/5 avg 0.973 (dos-fixed-qemu) | 1.13 |
| 27 | `MODEL_LEXICON_GOLD_V4_S3000` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | missing | PASS 5/5 avg 0.972 (float) | 1.13 |
| 28 | `MODEL_LEXICON_GOLD_V3_S3000` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | missing | PASS 5/5 avg 0.965 (float) | 1.12 |
| 29 | `MODEL_LEXICON4096_ADAPTED_W6S3000` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | NEEDS_TRAINING 4/5 avg 0.882 (dos-fixed-qemu) | NEEDS_TRAINING 4/5 avg 0.916 (dos-fixed-qemu) | 1.06 |
| 30 | `MODEL_LEXICON4096_W2S3000` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | missing | NEEDS_TRAINING 3/5 avg 0.773 (float) | 0.90 |
| 31 | `MODEL_LEXICON4096_ADAPTED_REPAIR2_S600` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | missing | NEEDS_TRAINING 0/5 avg 0.743 (float) | 0.86 |
| 32 | `MODEL_LEXICON4096_ADAPTED_REPAIR_S900` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | missing | NEEDS_TRAINING 2/5 avg 0.742 (float) | 0.86 |
| 33 | `MODEL_BYTE_BASELINE_PRE_LEXICON_PROMOTION` | `2L 48D 4H ctx192 h192 v258` | 90882 | 520740 | 5.42 model | missing | missing | missing | 0.00 |
| 34 | `MODEL_LEXICON_GOLD_V2_POSTTRAIN_MIX_V1_S120` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | missing | missing | 0.00 |
| 35 | `MODEL_LEXICON_GOLD_V2_POSTTRAIN_V1_S180` | `2L 48D 4H ctx192 h192 v4096` | 463168 | 2055940 | 2.27 model | missing | missing | missing | 0.00 |
| 36 | `MODEL_SUBWORD512_PROTO` | `2L 48D 4H ctx192 h192 v512` | 115520 | 622340 | 4.97 model | missing | missing | missing | 0.00 |

## Shape Coverage

| Shape | Checkpoints |
|---|---:|
| `2L 32D 4H ctx128 h128 v258` | 1 |
| `2L 48D 4H ctx192 h192 v258` | 17 |
| `2L 48D 4H ctx192 h192 v2710` | 1 |
| `2L 48D 4H ctx192 h192 v3768` | 1 |
| `2L 48D 4H ctx192 h192 v384` | 3 |
| `2L 48D 4H ctx192 h192 v4096` | 10 |
| `2L 48D 4H ctx192 h192 v512` | 1 |
| `3L 64D 4H ctx192 h256 v258` | 2 |

Current exports now cover multiple shapes. Use `architecture_profile_sweep.py` for the remaining trainer profiles and profile-level DOS evidence.
Active `MODEL` held-out source: `quality_report_dos_heldout.md` (dos-fixed-qemu).

## Gaps

- Missing held-out quality reports: MODEL_BYTE_BASELINE_PRE_LEXICON_PROMOTION, MODEL_LEXICON_GOLD_V2_POSTTRAIN_MIX_V1_S120, MODEL_LEXICON_GOLD_V2_POSTTRAIN_V1_S180, MODEL_SUBWORD512_PROTO
- Missing runtime-regression quality reports: MODEL_486DX2_DOMAIN_W2S1200, MODEL_BASELINE_PRE_FINETUNE1, MODEL_BASELINE_PRE_MPS, MODEL_BPE384_COMPLETE_W2S3000, MODEL_BPE384_DOMAIN_W2S3000, MODEL_BYTE_BASELINE_PRE_LEXICON_PROMOTION, MODEL_DOMAIN_ANCHOR_W2S1000, MODEL_DOMAIN_ANSWERBANK_W2S1000, MODEL_DOMAIN_CANDIDATE, MODEL_DOMAIN_FAMILY2_W2S1000, MODEL_DOMAIN_REHEARSAL, MODEL_DOMAIN_RETENTION_S250, MODEL_DOMAIN_W2S800, MODEL_LEXICON384_W2S3000, MODEL_LEXICON4096_ADAPTED_REPAIR2_S600, MODEL_LEXICON4096_ADAPTED_REPAIR_S900, MODEL_LEXICON4096_BOUNDARY_W32S3000, MODEL_LEXICON4096_W2S3000, MODEL_LEXICON_GOLD_V1_S3000, MODEL_LEXICON_GOLD_V2_POSTTRAIN_MIX_V1_S120, MODEL_LEXICON_GOLD_V2_POSTTRAIN_V1_S180, MODEL_LEXICON_GOLD_V3_S3000, MODEL_LEXICON_GOLD_V4_S3000, MODEL_ONLINE_CANDIDATE, MODEL_SUBWORD512_PROTO
- Missing direct DOS `--perf` per checkpoint: MODEL_486DX2_DOMAIN_W2S1200, MODEL_BASELINE_PRE_FINETUNE1, MODEL_BASELINE_PRE_MPS, MODEL_BPE384_COMPLETE_W2S3000, MODEL_BPE384_DOMAIN_W2S3000, MODEL_BYTE_BASELINE_PRE_LEXICON_PROMOTION, MODEL_CANDIDATE, MODEL_CANDIDATE_FINETUNE1, MODEL_CANDIDATE_MPS, MODEL_CANDIDATE_MPS2, MODEL_CANDIDATE_MPS3, MODEL_DOMAIN_ANCHOR_W2S1000, MODEL_DOMAIN_ANSWERBANK_W2S1000, MODEL_DOMAIN_CANDIDATE, MODEL_DOMAIN_FAMILY2_W2S1000, MODEL_DOMAIN_FAMILY_W2S1000, MODEL_DOMAIN_REHEARSAL, MODEL_DOMAIN_RETENTION_S250, MODEL_DOMAIN_W2S800, MODEL_LEXICON384_W2S3000, MODEL_LEXICON4096_ADAPTED_REPAIR2_S600, MODEL_LEXICON4096_ADAPTED_REPAIR_S900, MODEL_LEXICON4096_ADAPTED_W6S3000, MODEL_LEXICON4096_BOUNDARY_W32S3000, MODEL_LEXICON4096_W2S3000, MODEL_LEXICON_GOLD_V1_S3000, MODEL_LEXICON_GOLD_V2_POSTTRAIN_MIX_V1_S120, MODEL_LEXICON_GOLD_V2_POSTTRAIN_V1_S180, MODEL_LEXICON_GOLD_V2_S3000, MODEL_LEXICON_GOLD_V3_S3000, MODEL_LEXICON_GOLD_V4_S3000, MODEL_ONLINE_CANDIDATE, MODEL_SUBWORD512_PROTO

## Next Commands

```sh
python3 scripts/profile_pareto_report.py --refresh-heldout-float
bash qemu/run_perf_486.sh 386dx-33
bash qemu/run_perf_486.sh 486sx-25
bash qemu/run_perf_486.sh 486dx-33
bash qemu/run_perf_486.sh 486dx4-100
bash qemu/run_perf_486.sh pentium-60
bash qemu/run_perf_486.sh pentium-133
```

Evidence root: `/Users/tyr/Desktop/gpt2-basic/qemu/evidence`

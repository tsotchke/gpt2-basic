# Model Quality Repair Plan

The strict gate requires all-suite quality when available. Historical candidates that fail this gate should not be treated as latent release options.

## Summary

- `failed_repair`: 6
- `keep`: 2
- `keep_candidate`: 2
- `keep_release`: 6
- `reject_host_only`: 1
- `retire_superseded`: 22
- `retrain_experiment`: 2
- `retrain_optional`: 3
- `retrain_priority`: 2

## Actions

| Model | Quality | Decision | Reason |
|---|---|---|---|
| `MODEL` | PASS 10/10 avg 0.969 (dos-fixed-qemu, all) | `keep_release` | release-shaped model passes the strict all-suite gate |
| `MODEL_486DX2_DOMAIN_W2S1200` | NEEDS_TRAINING 0/10 avg 0.721 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_BASELINE_PRE_FINETUNE1` | NEEDS_TRAINING 2/10 avg 0.782 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_BASELINE_PRE_MPS` | NEEDS_TRAINING 1/10 avg 0.712 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_BPE384_COMPLETE_W2S3000` | NEEDS_TRAINING 0/10 avg 0.739 (float, all) | `retrain_experiment` | BPE path is an experiment; retrain only if comparing tokenizer families is still useful |
| `MODEL_BPE384_DOMAIN_W2S3000` | NEEDS_TRAINING 8/10 avg 0.794 (float, all) | `retrain_experiment` | BPE path is an experiment; retrain only if comparing tokenizer families is still useful |
| `MODEL_BYTE_BASELINE_PRE_LEXICON_PROMOTION` | NEEDS_TRAINING 2/10 avg 0.760 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_CANDIDATE` | NEEDS_TRAINING 1/10 avg 0.701 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_CANDIDATE_FINETUNE1` | NEEDS_TRAINING 2/10 avg 0.760 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_CANDIDATE_MPS` | NEEDS_TRAINING 2/10 avg 0.782 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_CANDIDATE_MPS2` | NEEDS_TRAINING 2/10 avg 0.713 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_CANDIDATE_MPS3` | NEEDS_TRAINING 2/10 avg 0.747 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_DOMAIN_ANCHOR_W2S1000` | NEEDS_TRAINING 0/10 avg 0.729 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_DOMAIN_ANSWERBANK_W2S1000` | NEEDS_TRAINING 2/10 avg 0.754 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_DOMAIN_CANDIDATE` | NEEDS_TRAINING 0/10 avg 0.687 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_DOMAIN_FAMILY2_W2S1000` | NEEDS_TRAINING 1/10 avg 0.752 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_DOMAIN_FAMILY_W2S1000` | NEEDS_TRAINING 7/10 avg 0.806 (dos-fixed-qemu, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_DOMAIN_REHEARSAL` | NEEDS_TRAINING 1/10 avg 0.717 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_DOMAIN_RETENTION_S250` | NEEDS_TRAINING 2/10 avg 0.692 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_DOMAIN_W2S800` | NEEDS_TRAINING 2/10 avg 0.736 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_HEADQ4_PROD_PROBE` | PASS 10/10 avg 0.960 (float, all) | `keep_release` | release-shaped model passes the strict all-suite gate |
| `MODEL_HEADSHORTLIST2048_PROD_PROBE` | PASS 10/10 avg 0.960 (float, all) | `keep_release` | release-shaped model passes the strict all-suite gate |
| `MODEL_LEXICON384_W2S3000` | NEEDS_TRAINING 2/10 avg 0.769 (float, all) | `retrain_optional` | fails strict all-suite but has no release-critical role |
| `MODEL_LEXICON4096_ADAPTED_REPAIR2_S600` | NEEDS_TRAINING 1/10 avg 0.768 (float, all) | `failed_repair` | repair attempt has artifact evidence but still fails strict all-suite quality |
| `MODEL_LEXICON4096_ADAPTED_REPAIR_S900` | NEEDS_TRAINING 7/10 avg 0.844 (float, all) | `failed_repair` | repair attempt has artifact evidence but still fails strict all-suite quality |
| `MODEL_LEXICON4096_ADAPTED_W6S3000` | NEEDS_TRAINING 8/10 avg 0.899 (dos-fixed-qemu, all) | `retire_superseded` | adapted/repair branch is superseded by the gold-v4 release path |
| `MODEL_LEXICON4096_BOUNDARY_W32S3000` | NEEDS_TRAINING 1/10 avg 0.810 (float, all) | `retrain_optional` | fails strict all-suite but has no release-critical role |
| `MODEL_LEXICON4096_W2S3000` | NEEDS_TRAINING 6/10 avg 0.829 (float, all) | `retrain_optional` | fails strict all-suite but has no release-critical role |
| `MODEL_LEXICON_GOLD_V1_S3000` | PASS 10/10 avg 0.890 (float, all) | `keep_candidate` | candidate passes host strict all-suite; needs DOS vector/perf evidence before promotion |
| `MODEL_LEXICON_GOLD_V2_POSTTRAIN_MIX_V1_S120` | NEEDS_TRAINING 7/10 avg 0.964 (float, all) | `retire_superseded` | post-training repair damaged fluency; keep as negative evidence only |
| `MODEL_LEXICON_GOLD_V2_POSTTRAIN_V1_S180` | NEEDS_TRAINING 3/10 avg 0.877 (float, all) | `retire_superseded` | post-training repair damaged fluency; keep as negative evidence only |
| `MODEL_LEXICON_GOLD_V2_S3000` | PASS 10/10 avg 0.961 (dos-fixed-qemu, all) | `retire_superseded` | gold-v2 is quality-proven but superseded by gold-v4 default evidence |
| `MODEL_LEXICON_GOLD_V3_S3000` | PASS 10/10 avg 0.930 (float, all) | `keep_candidate` | candidate passes host strict all-suite; needs DOS vector/perf evidence before promotion |
| `MODEL_LEXICON_GOLD_V4_S3000` | PASS 10/10 avg 0.969 (dos-fixed-qemu, all) | `keep_release` | release-shaped model passes the strict all-suite gate |
| `MODEL_ONLINE_CANDIDATE` | NEEDS_TRAINING 0/10 avg 0.670 (float, all) | `retire_superseded` | old byte/domain candidate fails strict all-suite and is superseded by lexicon-gold models |
| `MODEL_PROFILE_386_MIN` | NEEDS_TRAINING 1/10 avg 0.612 (dos-fixed-qemu, all) | `retrain_priority` | fastest measured profile is valuable only if a small-vocabulary repair passes quality |
| `MODEL_PROFILE_386_MIN_LEXICON384_REPAIR` | NEEDS_TRAINING 2/10 avg 0.714 (float, all) | `failed_repair` | repair attempt has artifact evidence but still fails strict all-suite quality |
| `MODEL_PROFILE_486DX2_CACHEFOCUS_REPAIR` | NEEDS_TRAINING 6/10 avg 0.805 (float, all) | `failed_repair` | repair attempt has artifact evidence but still fails strict all-suite quality |
| `MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR` | NEEDS_TRAINING 9/10 avg 0.916 (float, all) | `failed_repair` | repair attempt has artifact evidence but still fails strict all-suite quality |
| `MODEL_PROFILE_486DX2_LEXICON4096_REPAIR` | NEEDS_TRAINING 9/10 avg 0.894 (float, all) | `failed_repair` | repair attempt has artifact evidence but still fails strict all-suite quality |
| `MODEL_PROFILE_486DX2_USABLE` | NEEDS_TRAINING 2/10 avg 0.715 (dos-fixed-qemu, all) | `retrain_priority` | larger 486DX2 profile should be retrained from the clean gold curriculum, not old byte data |
| `MODEL_SUBWORD512_PROTO` | NEEDS_TRAINING 0/5 avg 0.166 (subword-host, heldout) | `reject_host_only` | host-only tokenizer prototype is not DOS-ready and fails quality |
| `MODEL_TOKHEADQ4_PROD_PROBE` | PASS 10/10 avg 0.960 (float, all) | `keep_release` | release-shaped model passes the strict all-suite gate |
| `MODEL_TOKHEADQ4_STREAM_PROD_PROBE` | PASS 10/10 avg 0.960 (float, all) | `keep_release` | release-shaped model passes the strict all-suite gate |
| `ASSISTANT_DOSHELP` | PASS 4/4 avg 0.960 (float, assistant-pack) | `keep` | assistant pack model passes its pack-local strict gate |
| `ASSISTANT_OFFICE` | PASS 4/4 avg 0.972 (float, assistant-pack) | `keep` | assistant pack model passes its pack-local strict gate |

## Retrain Commands

```sh
python3 scripts/train_tiny_gpt.py --profile 486sx-safe --tokenizer bpe --vocab-size 384 --include-docs --corpus-file data/domain_curriculum/gold_curriculum_v4.txt --corpus-weight 4 --steps 5000 --log-every 500 --output assets/gpt2_basic/MODEL_BPE384_GOLD_V4_REPAIR
```

```sh
python3 scripts/evaluate_gpt2_basic_quality.py --model-dir assets/gpt2_basic/MODEL_BPE384_GOLD_V4_REPAIR --suite all --backend float --threshold 0.72 --output qemu/evidence/quality_report_bpe384_gold_v4_repair_all.md
```

```sh
python3 scripts/train_tiny_gpt.py --profile 386-min --tokenizer lexicon --vocab-size 384 --include-docs --corpus-file data/domain_curriculum/gold_curriculum_v4.txt --corpus-weight 4 --steps 5000 --log-every 500 --output assets/gpt2_basic/MODEL_PROFILE_386_MIN_LEXICON384_REPAIR
```

```sh
python3 scripts/evaluate_gpt2_basic_quality.py --model-dir assets/gpt2_basic/MODEL_PROFILE_386_MIN_LEXICON384_REPAIR --suite all --backend float --threshold 0.72 --output qemu/evidence/quality_report_profile_386_min_lexicon384_repair_all.md
```

```sh
python3 scripts/train_tiny_gpt.py --profile 486dx2-usable --tokenizer lexicon --vocab-size 4096 --lexicon-min-count 1 --base-weight 0 --corpus-file data/domain_curriculum/gold_curriculum_v5_clean_repair.txt --corpus-weight 1 --steps 7000 --log-every 500 --output assets/gpt2_basic/MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR2
```

```sh
python3 scripts/evaluate_gpt2_basic_quality.py --model-dir assets/gpt2_basic/MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR2 --suite all --backend float --threshold 0.72 --output qemu/evidence/quality_report_profile_486dx2_cleanlexicon4096_repair2_all.md
```

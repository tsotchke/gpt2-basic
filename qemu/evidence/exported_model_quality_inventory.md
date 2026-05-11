# Exported Model Quality Inventory

Models audited: `46`
Artifact pass: `46/46`
Quality pass: `11/46`
Needs training: `35`
Missing quality evidence: `0`

## Models

| Role | Model | Shape | Artifacts | Best Quality | Reports |
|---|---|---|---|---|---|
| assistant_pack | `ASSISTANT_DOSHELP` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 4/4 avg 0.954 (float, assistant-pack) | `quality_report_assistant_doshelp.md` |
| assistant_pack | `ASSISTANT_OFFICE` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 4/4 avg 0.966 (float, assistant-pack) | `quality_report_assistant_office.md` |
| root | `MODEL` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 10/10 avg 0.969 (dos-fixed-qemu, all) | `quality_report_dos_heldout.md`, `quality_report_dos_all.md`, `quality_report_default_model_all.md`, `+2 more` |
| root | `MODEL_486DX2_DOMAIN_W2S1200` | `3L 64D 4H ctx192 h256 v258` | PASS | NEEDS_TRAINING 0/10 avg 0.721 (float, all) | `quality_report_486dx2_domain_w2s1200_runtime.md`, `quality_report_486dx2_domain_w2s1200_all.md`, `quality_report_486dx2_domain_w2s1200_heldout.md` |
| root | `MODEL_BASELINE_PRE_FINETUNE1` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.782 (float, all) | `quality_report_baseline_pre_finetune1_heldout.md`, `quality_report_baseline_pre_finetune1_all.md` |
| root | `MODEL_BASELINE_PRE_MPS` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 1/10 avg 0.712 (float, all) | `quality_report_baseline_pre_mps_all.md`, `quality_report_baseline_pre_mps_heldout.md` |
| root | `MODEL_BPE384_COMPLETE_W2S3000` | `2L 48D 4H ctx192 h192 v384` | PASS | NEEDS_TRAINING 0/10 avg 0.739 (float, all) | `quality_report_bpe384_complete_w2s3000_heldout.md`, `quality_report_bpe384_complete_w2s3000_runtime.md`, `quality_report_bpe384_complete_w2s3000_all.md` |
| root | `MODEL_BPE384_DOMAIN_W2S3000` | `2L 48D 4H ctx192 h192 v384` | PASS | NEEDS_TRAINING 8/10 avg 0.794 (float, all) | `quality_report_bpe384_domain_w2s3000_all.md`, `quality_report_bpe384_domain_w2s3000_runtime.md`, `quality_report_bpe384_domain_w2s3000_heldout.md` |
| root | `MODEL_BYTE_BASELINE_PRE_LEXICON_PROMOTION` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.760 (float, all) | `quality_report_byte_baseline_pre_lexicon_promotion_all.md` |
| root | `MODEL_CANDIDATE` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 1/10 avg 0.701 (float, all) | `quality_report_candidate.md`, `quality_report_candidate_all.md`, `quality_report_candidate_heldout.md` |
| root | `MODEL_CANDIDATE_FINETUNE1` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.760 (float, all) | `quality_report_candidate_finetune1.md`, `quality_report_candidate_finetune1_heldout.md`, `quality_report_candidate_finetune1_all.md` |
| root | `MODEL_CANDIDATE_MPS` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.782 (float, all) | `quality_report_candidate_mps.md`, `quality_report_candidate_mps_heldout.md`, `quality_report_candidate_mps_all.md` |
| root | `MODEL_CANDIDATE_MPS2` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.713 (float, all) | `quality_report_candidate_mps2.md`, `quality_report_candidate_mps2_all.md`, `quality_report_candidate_mps2_heldout.md` |
| root | `MODEL_CANDIDATE_MPS3` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.747 (float, all) | `quality_report_candidate_mps3.md`, `quality_report_candidate_mps3_all.md`, `quality_report_candidate_mps3_heldout.md` |
| root | `MODEL_DOMAIN_ANCHOR_W2S1000` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 0/10 avg 0.729 (float, all) | `quality_report_domain_anchor_w2s1000_runtime.md`, `quality_report_domain_anchor_w2s1000_heldout.md`, `quality_report_domain_anchor_w2s1000_all.md` |
| root | `MODEL_DOMAIN_ANSWERBANK_W2S1000` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.754 (float, all) | `quality_report_domain_answerbank_w2s1000_runtime.md`, `quality_report_domain_answerbank_w2s1000_all.md`, `quality_report_domain_answerbank_w2s1000_heldout.md` |
| root | `MODEL_DOMAIN_CANDIDATE` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 0/10 avg 0.687 (float, all) | `quality_report_domain_candidate_heldout.md`, `quality_report_domain_candidate_all.md`, `quality_report_domain_candidate_runtime.md` |
| root | `MODEL_DOMAIN_FAMILY2_W2S1000` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 1/10 avg 0.752 (float, all) | `quality_report_domain_family2_w2s1000_runtime.md`, `quality_report_domain_family2_w2s1000_all.md`, `quality_report_domain_family2_w2s1000_heldout.md` |
| root | `MODEL_DOMAIN_FAMILY_W2S1000` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 7/10 avg 0.806 (dos-fixed-qemu, all) | `quality_report_domain_family_w2s1000_heldout.md`, `quality_report_dos_model_domain_family_w2s1000_heldout.md`, `quality_report_dos_model_domain_family_w2s1000_all.md`, `+3 more` |
| root | `MODEL_DOMAIN_REHEARSAL` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 1/10 avg 0.717 (float, all) | `quality_report_domain_rehearsal_runtime.md`, `quality_report_domain_rehearsal_all.md`, `quality_report_domain_rehearsal_heldout.md` |
| root | `MODEL_DOMAIN_RETENTION_S250` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.692 (float, all) | `quality_report_domain_retention_s250_runtime.md`, `quality_report_domain_retention_s250_heldout.md`, `quality_report_domain_retention_s250_all.md` |
| root | `MODEL_DOMAIN_W2S800` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.736 (float, all) | `quality_report_domain_w2s800_runtime.md`, `quality_report_domain_w2s800_all.md`, `quality_report_domain_w2s800_heldout.md` |
| root | `MODEL_HEADQ4_PROD_PROBE` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 10/10 avg 0.960 (float, all) | `quality_report_headq4_prod_probe_heldout.md`, `quality_report_headq4_prod_probe_all.md`, `quality_report_headq4_prod_probe_fixed_all.md`, `+1 more` |
| root | `MODEL_HEADSHORTLIST2048_PROD_PROBE` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 10/10 avg 0.960 (float, all) | `quality_report_headshortlist2048_heldout.md`, `quality_report_headshortlist2048_prod_probe_all.md`, `quality_report_headshortlist2048_runtime.md` |
| root | `MODEL_LEXICON384_W2S3000` | `2L 48D 4H ctx192 h192 v384` | PASS | NEEDS_TRAINING 2/10 avg 0.769 (float, all) | `quality_report_lexicon384_w2s3000_all.md`, `quality_report_lexicon384_w2s3000_runtime.md`, `quality_report_lexicon384_w2s3000_heldout.md` |
| root | `MODEL_LEXICON4096_ADAPTED_REPAIR2_S600` | `2L 48D 4H ctx192 h192 v4096` | PASS | NEEDS_TRAINING 1/10 avg 0.768 (float, all) | `quality_report_lexicon4096_adapted_repair2_s600_runtime.md`, `quality_report_lexicon4096_adapted_repair2_s600_all.md`, `quality_report_lexicon4096_adapted_repair2_s600_heldout.md` |
| root | `MODEL_LEXICON4096_ADAPTED_REPAIR_S900` | `2L 48D 4H ctx192 h192 v4096` | PASS | NEEDS_TRAINING 7/10 avg 0.844 (float, all) | `quality_report_lexicon4096_adapted_repair_s900_runtime.md`, `quality_report_lexicon4096_adapted_repair_s900_all.md`, `quality_report_lexicon4096_adapted_repair_s900_heldout.md` |
| root | `MODEL_LEXICON4096_ADAPTED_W6S3000` | `2L 48D 4H ctx192 h192 v4096` | PASS | NEEDS_TRAINING 8/10 avg 0.899 (dos-fixed-qemu, all) | `quality_report_lexicon4096_adapted_w6s3000_heldout.md`, `quality_report_lexicon4096_adapted_w6s3000_all.md`, `quality_report_dos_model_lexicon4096_adapted_w6s3000_heldout.md`, `+3 more` |
| root | `MODEL_LEXICON4096_BOUNDARY_W32S3000` | `2L 48D 4H ctx192 h192 v3768` | PASS | NEEDS_TRAINING 1/10 avg 0.810 (float, all) | `quality_report_lexicon4096_boundary_w32s3000_runtime.md`, `quality_report_lexicon4096_boundary_w32s3000_heldout.md`, `quality_report_lexicon4096_boundary_w32s3000_all.md` |
| root | `MODEL_LEXICON4096_W2S3000` | `2L 48D 4H ctx192 h192 v4096` | PASS | NEEDS_TRAINING 6/10 avg 0.829 (float, all) | `quality_report_lexicon4096_w2s3000_runtime.md`, `quality_report_lexicon4096_w2s3000_all.md`, `quality_report_lexicon4096_w2s3000_heldout.md` |
| root | `MODEL_LEXICON_GOLD_V1_S3000` | `2L 48D 4H ctx192 h192 v2710` | PASS | PASS 10/10 avg 0.890 (float, all) | `quality_report_lexicon_gold_v1_s3000_all.md`, `quality_report_lexicon_gold_v1_s3000_heldout.md`, `quality_report_lexicon_gold_v1_s3000_runtime.md` |
| root | `MODEL_LEXICON_GOLD_V2_POSTTRAIN_MIX_V1_S120` | `2L 48D 4H ctx192 h192 v4096` | PASS | NEEDS_TRAINING 7/10 avg 0.964 (float, all) | `quality_report_lexicon_gold_v2_posttrain_mix_v1_s120_all.md` |
| root | `MODEL_LEXICON_GOLD_V2_POSTTRAIN_V1_S180` | `2L 48D 4H ctx192 h192 v4096` | PASS | NEEDS_TRAINING 3/10 avg 0.877 (float, all) | `quality_report_lexicon_gold_v2_posttrain_v1_s180_all.md` |
| root | `MODEL_LEXICON_GOLD_V2_S3000` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 10/10 avg 0.961 (dos-fixed-qemu, all) | `quality_report_dos_model_lexicon_gold_v2_s3000_heldout.md`, `quality_report_lexicon_gold_v2_s3000_heldout_stopfix.md`, `quality_report_dos_model_lexicon_gold_v2_s3000_all.md`, `+8 more` |
| root | `MODEL_LEXICON_GOLD_V3_S3000` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 10/10 avg 0.930 (float, all) | `quality_report_lexicon_gold_v3_s3000_heldout.md`, `quality_report_lexicon_gold_v3_s3000_all.md`, `quality_report_lexicon_gold_v3_s3000_runtime.md` |
| root | `MODEL_LEXICON_GOLD_V4_S3000` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 10/10 avg 0.969 (dos-fixed-qemu, all) | `quality_report_dos_model_lexicon_gold_v4_s3000_heldout.md`, `quality_report_lexicon_gold_v4_s3000_heldout.md`, `quality_report_dos_model_lexicon_gold_v4_s3000_all.md`, `+4 more` |
| root | `MODEL_ONLINE_CANDIDATE` | `2L 48D 4H ctx192 h192 v258` | PASS | NEEDS_TRAINING 0/10 avg 0.670 (float, all) | `quality_report_online_candidate_runtime.md`, `quality_report_online_candidate_all.md`, `quality_report_online_candidate_heldout.md` |
| root | `MODEL_PROFILE_386_MIN` | `2L 32D 4H ctx128 h128 v258` | PASS | NEEDS_TRAINING 1/10 avg 0.612 (dos-fixed-qemu, all) | `quality_report_dos_model_profile_386_min.md`, `quality_report_profile_386_min_all.md`, `quality_report_dos_model_profile_386_min_all.md`, `+2 more` |
| root | `MODEL_PROFILE_386_MIN_LEXICON384_REPAIR` | `2L 32D 4H ctx128 h128 v384` | PASS | NEEDS_TRAINING 2/10 avg 0.714 (float, all) | `quality_report_profile_386_min_lexicon384_repair_all.md` |
| root | `MODEL_PROFILE_486DX2_CACHEFOCUS_REPAIR` | `3L 64D 4H ctx192 h256 v4096` | PASS | NEEDS_TRAINING 6/10 avg 0.805 (float, all) | `quality_report_profile_486dx2_cachefocus_repair_all.md` |
| root | `MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR` | `3L 64D 4H ctx192 h256 v4096` | PASS | NEEDS_TRAINING 9/10 avg 0.916 (float, all) | `quality_report_profile_486dx2_cleanlexicon4096_repair_all.md` |
| root | `MODEL_PROFILE_486DX2_LEXICON4096_REPAIR` | `3L 64D 4H ctx192 h256 v2686` | PASS | NEEDS_TRAINING 9/10 avg 0.894 (float, all) | `quality_report_profile_486dx2_lexicon4096_repair_all.md` |
| root | `MODEL_PROFILE_486DX2_USABLE` | `3L 64D 4H ctx192 h256 v258` | PASS | NEEDS_TRAINING 2/10 avg 0.715 (dos-fixed-qemu, all) | `quality_report_dos_model_profile_486dx2_usable.md`, `quality_report_dos_model_profile_486dx2_usable_all.md`, `quality_report_profile_486dx2_usable_all.md`, `+2 more` |
| root | `MODEL_SUBWORD512_PROTO` | `2L 48D 4H ctx192 h192 v512` | PASS | NEEDS_TRAINING 0/5 avg 0.166 (subword-host, heldout) | `quality_heldout.md`, `quality_runtime.md` |
| root | `MODEL_TOKHEADQ4_PROD_PROBE` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 10/10 avg 0.960 (float, all) | `quality_report_tokheadq4_prod_probe_heldout.md`, `quality_report_tokheadq4_prod_probe_all.md`, `quality_report_tokheadq4_prod_probe_fixed_all.md`, `+1 more` |
| root | `MODEL_TOKHEADQ4_STREAM_PROD_PROBE` | `2L 48D 4H ctx192 h192 v4096` | PASS | PASS 10/10 avg 0.960 (float, all) | `quality_report_tokheadq4_stream_prod_probe_all.md` |

## Gaps

- `MODEL_486DX2_DOMAIN_W2S1200`: quality needs training
- `MODEL_BASELINE_PRE_FINETUNE1`: quality needs training
- `MODEL_BASELINE_PRE_MPS`: quality needs training
- `MODEL_BPE384_COMPLETE_W2S3000`: quality needs training
- `MODEL_BPE384_DOMAIN_W2S3000`: quality needs training
- `MODEL_BYTE_BASELINE_PRE_LEXICON_PROMOTION`: quality needs training
- `MODEL_CANDIDATE`: quality needs training
- `MODEL_CANDIDATE_FINETUNE1`: quality needs training
- `MODEL_CANDIDATE_MPS`: quality needs training
- `MODEL_CANDIDATE_MPS2`: quality needs training
- `MODEL_CANDIDATE_MPS3`: quality needs training
- `MODEL_DOMAIN_ANCHOR_W2S1000`: quality needs training
- `MODEL_DOMAIN_ANSWERBANK_W2S1000`: quality needs training
- `MODEL_DOMAIN_CANDIDATE`: quality needs training
- `MODEL_DOMAIN_FAMILY2_W2S1000`: quality needs training
- `MODEL_DOMAIN_FAMILY_W2S1000`: quality needs training
- `MODEL_DOMAIN_REHEARSAL`: quality needs training
- `MODEL_DOMAIN_RETENTION_S250`: quality needs training
- `MODEL_DOMAIN_W2S800`: quality needs training
- `MODEL_LEXICON384_W2S3000`: quality needs training
- `MODEL_LEXICON4096_ADAPTED_REPAIR2_S600`: quality needs training
- `MODEL_LEXICON4096_ADAPTED_REPAIR_S900`: quality needs training
- `MODEL_LEXICON4096_ADAPTED_W6S3000`: quality needs training
- `MODEL_LEXICON4096_BOUNDARY_W32S3000`: quality needs training
- `MODEL_LEXICON4096_W2S3000`: quality needs training
- `MODEL_LEXICON_GOLD_V2_POSTTRAIN_MIX_V1_S120`: quality needs training
- `MODEL_LEXICON_GOLD_V2_POSTTRAIN_V1_S180`: quality needs training
- `MODEL_ONLINE_CANDIDATE`: quality needs training
- `MODEL_PROFILE_386_MIN`: quality needs training
- `MODEL_PROFILE_386_MIN_LEXICON384_REPAIR`: quality needs training
- `MODEL_PROFILE_486DX2_CACHEFOCUS_REPAIR`: quality needs training
- `MODEL_PROFILE_486DX2_CLEANLEXICON4096_REPAIR`: quality needs training
- `MODEL_PROFILE_486DX2_LEXICON4096_REPAIR`: quality needs training
- `MODEL_PROFILE_486DX2_USABLE`: quality needs training
- `MODEL_SUBWORD512_PROTO`: quality needs training

Evidence root: `/Users/tyr/Desktop/gpt2-basic/qemu/evidence`

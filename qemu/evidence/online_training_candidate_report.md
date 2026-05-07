# GPT2-BASIC Online Corpus Candidate Result

Date: 2026-05-06

## Candidate

- Model directory: `assets/gpt2_basic/MODEL_ONLINE_CANDIDATE`
- Profile: `486sx-safe`
- Training command: `.venv-torch311/bin/python scripts/train_gpt2_basic.py --profile 486sx-safe --include-docs --corpus-file data/online_corpus/online_training_corpus.txt --corpus-weight 1 --device mps --steps 2500 --sample-tokens 160 --log-every 250 --output assets/gpt2_basic/MODEL_ONLINE_CANDIDATE`
- Training tokens: 18,025,830
- Training docs: 46 core/repo docs, 407 external corpus chunks
- Final observed loss: 1.3785
- Artifact status: `scripts/model_report.py --model-dir assets/gpt2_basic/MODEL_ONLINE_CANDIDATE --strict` returned OK

## Quality Evidence

| Check | Backend | Status | Pass rate | Average |
|---|---|---|---:|---:|
| Held-out suite | host float / MPS | NEEDS_TRAINING | 0/5 | 0.650 |
| Runtime-regression suite | host float / MPS | NEEDS_TRAINING | 0/5 | 0.678 |

Reports:

- `qemu/evidence/quality_report_online_candidate_heldout.md`
- `qemu/evidence/quality_report_online_candidate_runtime.md`

## Finding

The conservative online corpus improves raw language-like character flow, but the
candidate is worse than the active `486sx-safe` DOS held-out baseline
(`0.685`). It should not be staged to DOS as a replacement.

The failure mode is expected for a tiny byte-level model trained from scratch on
mostly generic public-domain fiction: it learns common word fragments and old
prose rhythm, while diluting the specific runtime vocabulary needed for cache,
timing, fixed-point arithmetic, model profiles, and DOS execution.

## Next Data Architecture

Use the online corpus as pretraining material, not as the final training mix.
The production training plan should be:

1. Pretrain on conservative permissive online prose for English continuity.
2. Add opt-in domain sources only when obligations are explicit, especially
   FreeDOS documentation if ShareAlike terms are acceptable.
3. Fine-tune on project/runtime documents and new held-out-adjacent domain
   explanations that do not copy the evaluation prompts.
4. Promote a checkpoint to QEMU only after host held-out quality exceeds the
   active DOS baseline.

This keeps data quality tied to the real production target instead of chasing
corpus size.

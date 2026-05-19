# GPT2-BASIC Adapted Curriculum

This corpus adapts measured project results into tokenizer-aware training text.
It keeps clean domain language and filters path/config noise before large-vocabulary training.

## Summary

- Corpus: `data/domain_curriculum/adapted_boundary_curriculum.txt`
- Documents: 279
- Characters: 51592
- Seed repeats: 6
- Seed docs before filtering: 1614
- Mined docs before filtering: 264
- Retained docs: 279

## Sources

- `data/domain_curriculum/domain_curriculum.txt`: 244 retained candidate spans before dedupe
- `qemu/evidence/quality_report_lexicon4096_w2s3000_runtime.md`: 1 retained candidate spans before dedupe
- `qemu/evidence/quality_report_lexicon4096_w2s3000_heldout.md`: 4 retained candidate spans before dedupe
- `qemu/evidence/quality_report_bpe384_complete_w2s3000_heldout.md`: 5 retained candidate spans before dedupe
- `qemu/evidence/quality_report_bpe384_complete_w2s3000_runtime.md`: 3 retained candidate spans before dedupe
- `qemu/evidence/quality_report_domain_family_w2s1000_heldout.md`: 1 retained candidate spans before dedupe
- `qemu/evidence/quality_report_domain_family_w2s1000_runtime.md`: 2 retained candidate spans before dedupe
- `qemu/evidence/quality_report_dos_model_domain_family_w2s1000_heldout.md`: 1 retained candidate spans before dedupe
- `qemu/evidence/quality_report_dos_model_domain_family_w2s1000.md`: 3 retained candidate spans before dedupe

## Training Intent

- Keep a large vocabulary useful by feeding it complete, reusable pieces.
- Avoid making filenames, commands, and one-off report artifacts likely output tokens.
- Preserve the best fixed-point, cache, timing, profile, runtime, and real-inference wording.

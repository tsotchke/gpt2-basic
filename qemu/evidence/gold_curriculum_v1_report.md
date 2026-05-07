# GPT2-BASIC Gold Curriculum v1

This is a hand-curated narrow corpus for the large lexicon tokenizer path.
It avoids report mining, scoring instructions, code paths, and previous malformed generations.

## Summary

- Corpus: `data/domain_curriculum/gold_curriculum_v1.txt`
- Documents: 154
- Characters: 37616
- Audit failures: 0
- Probe lexicon vocab size: 2657
- Probe lexicon targets: 3062
- Probe alphabetic byte targets: 16
- Probe alphabetic byte share: 0.005

## Top Probe Vocabulary Pieces

- ` the`
- ` and`
- ` model`
- ` memory`
- ` runtime`
- ` The`
- ` transformer`
- ` checkpoint`
- ` attention`
- ` context length`
- ` the model`
- ` context`
- ` tokens`
- ` cache`
- ` output`
- ` vocabulary`
- ` is`
- ` arithmetic`
- ` the DOS`
- ` quality`
- ` fixed-point`
- ` attention work`
- ` GPT2-BASIC`
- ` tokens per`
- ` because`
- ` length`
- ` cache memory`
- ` logits`
- ` weights`
- ` comes from`
- ` timing`
- ` per second`
- ` generated`
- ` inference`
- ` profile`
- ` the checkpoint`
- ` generation`
- ` belong together`
- ` in this system.`
- ` engineering`
- ` not`
- ` useful`
- ` from`
- ` an engineering`
- ` pressure comes`
- ` DOS`
- ` model can`
- ` with`
- ` can`
- ` when`
- ` paragraph can`
- ` the model can`
- ` integer`
- ` predictable`
- ` evidence`
- ` instead of`
- ` machine can run`
- ` model files are`
- ` layers`
- ` this system.`
- ` path uses the`
- ` work buffers.`
- ` keeps`
- ` needs`
- ` generated text.`
- ` technical words`
- ` can connect`
- ` together in`
- ` feed-forward`
- ` and output`
- ` BASIC runtime`
- ` checkpoint is`
- ` the DOS model`
- `For GPT2-BASIC`
- ` and the`
- ` without`
- ` evidence.`
- ` technical`
- ` implementation`
- ` matter because`

## Training Result

Candidate: `assets/gpt2_basic/MODEL_LEXICON_GOLD_V1_S3000`

- Training documents after packing: 35
- Actual trained vocab: 2710
- Parameters: 328,726
- Fixed weights: 1,314,904 bytes
- `VOCAB.BIN`: 67,774 bytes
- Host held-out strict score: 2/5 average 0.923
- Host runtime-regression strict score: 4/5 average 0.893
- DOS vector parity: 3/3 vectors, 39/39 phases
- DOS runtime memory: 1,501,540 bytes

Decision: reject as a promotion candidate. The corpus is clean enough to reduce
malformed byte-built words, but it is not broad enough to cover all held-out
prompts without repetition and topic drift. Use this as the seed for a larger
gold curriculum v2.

# GPT2-BASIC Gold Curriculum v2

This is a hand-curated narrow corpus for the large lexicon tokenizer path.
It avoids report mining, scoring instructions, code paths, and previous malformed generations.

## Summary

- Corpus: `data/domain_curriculum/gold_curriculum_v2.txt`
- Documents: 351
- Characters: 80540
- Audit failures: 0
- Probe lexicon vocab size: 4096
- Probe lexicon targets: 6528
- Probe alphabetic byte targets: 170
- Probe alphabetic byte share: 0.022

## Top Probe Vocabulary Pieces

- ` the`
- ` and`
- ` model`
- ` memory`
- ` transformer`
- ` checkpoint`
- ` runtime`
- ` The`
- ` context length`
- ` attention`
- ` the model`
- ` context`
- ` tokens`
- ` vocabulary`
- ` cache`
- ` is`
- ` output`
- ` fixed-point`
- ` quality`
- ` arithmetic`
- ` because`
- ` timing`
- ` tokens per`
- ` the checkpoint`
- ` inference`
- ` length`
- ` useful`
- ` generated`
- ` cache memory`
- ` logits`
- ` the DOS`
- ` weights`
- ` not`
- ` profile`
- ` attention work`
- ` comes from`
- ` per second`
- ` from`
- ` with`
- ` predictable`
- ` when`
- ` generation`
- ` evidence`
- ` instead of`
- ` DOS`
- ` can`
- ` layers`
- ` model can`
- ` the model can`
- ` model files are`
- ` implementation`
- ` pressure comes`
- ` integer`
- ` without`
- ` checkpoint is`
- ` work buffers.`
- ` the prompt`
- ` generated text.`
- ` machine can run`
- ` technical words`
- ` technical`
- ` and the`
- ` vectors`
- ` hardware`
- ` the host`
- ` needs`
- ` arrays`
- ` evidence.`
- ` floating point`
- ` transformer on`
- ` path uses the`
- ` timing result`
- ` keeps`
- ` engineering`
- ` model files`
- ` GPT2-BASIC`
- ` feed-forward`
- ` matter because`
- ` required model`
- ` is not`

## Training Result

Candidate: `assets/gpt2_basic/MODEL_LEXICON_GOLD_V2_S3000`
Promoted default: `assets/gpt2_basic/MODEL`

- Packed training documents: 122
- Actual trained vocab: 4096
- Output-allowed tokens: 3934
- Parameters: 463,168
- Fixed weights: 1,852,672 bytes
- `VOCAB.BIN`: 102,424 bytes
- Model directory: about 3.7 MB
- Host held-out strict score after lexicon-aware stop rule and boundary-audit correction: 5/5 average 0.973
- Host runtime-regression strict score after 30-token lexicon-aware stop rule and boundary-audit correction: 5/5 average 0.932
- Host all-suite strict score: 10/10 average 0.952
- Host all-suite with prompt-aware starter prior: 10/10 average 0.960
- Host fixed all-suite with prompt-aware starter prior: 10/10 average 0.960
- DOS held-out score after lexicon-aware stop rule and boundary-audit correction: 5/5 average 0.973
- DOS runtime-regression score after 30-token lexicon-aware stop rule and boundary-audit correction: 5/5 average 0.971
- DOS all-suite score after 30-token lexicon-aware stop rule and boundary-audit correction: 10/10 average 0.972
- DOS held-out score with prompt-aware starter prior: 5/5 average 0.973
- DOS runtime-regression score with prompt-aware starter prior: 5/5 average 0.948
- DOS all-suite score with prompt-aware starter prior: 10/10 average 0.961
- DOS vector parity: 3/3 vectors, 39/39 phases
- DOS runtime memory: 2,055,940 bytes

Decision: promote as the default checkpoint. The corpus fills the 4096-token
ceiling with clean domain pieces and keeps word formation much cleaner than the
polluted adapted corpora. The 30-token lexicon-aware stop rule fixed most long
repeated tails, and the corrected boundary audit no longer rejects the valid
word `context`. A narrow prompt-aware starter prior now fixes the worst
adjacent-fragment starts without changing weights: `DOS language models need`
starts with a compact-checkpoint answer, `A BASIC transformer runtime` starts
with a runtime verb, and `To improve performance on real hardware` starts with
an action phrase. Remaining work is broader prompt coverage and product polish,
not proving that the large-vocabulary DOS path is viable.

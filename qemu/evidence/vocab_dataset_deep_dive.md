# GPT2-BASIC Vocabulary And Dataset Deep Dive

Date: 2026-05-06

## Current Position

The promoted default is now `assets/gpt2_basic/MODEL`, copied from
`assets/gpt2_basic/MODEL_LEXICON_GOLD_V2_S3000` with the prompt-aware starter
prior in the host and DOS samplers.

- Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
- Parameters: 463,168
- Fixed weights: 1,852,672 bytes
- `VOCAB.BIN`: 102,424 bytes
- DOS runtime memory: 2,055,940 bytes
- DOS peak memory: about 1.96 MB
- Host float all-suite after prompt-aware starter prior: 10/10 average 0.960
- Host fixed all-suite after prompt-aware starter prior: 10/10 average 0.960
- DOS all-suite after prompt-aware starter prior: 10/10 average 0.961
- DOS vector parity: 3/3 vectors, 39/39 phases, `VECTOR_CHECK_OK`

The large-vocabulary DOS path is now viable. The remaining work is not raw
vocabulary size; it is prompt coverage, answer boundaries, and keeping future
training data free of scaffolding or generated failure text.

## Vocabulary Contents

The 4096-token lexicon is mostly space-prefixed word and phrase pieces. The
malformed words seen in completions, such as `saloing`, `satizimblang`,
`cheful`, `plort`, `atthion`, `singht`, and `andul`, are not stored as whole
lexicon tokens. They encode as individual byte tokens.

That means the failure is a target-distribution problem: the model is learning
to continue alphabetic byte runs when it falls off the lexicon path.

The highest-ranked lexicon pieces reveal the dataset problem. Early pieces in
`MODEL_LEXICON4096_ADAPTED_W6S3000/VOCAB.BIN` include:

- ` answer should`
- ` then tie them`
- ` and then tie`
- ` tie them to`
- ` should be short`
- ` should connect`
- ` depends on this`
- ` practical rule`
- ` should say that`
- ` correct answer`
- ` target behavior`
- ` relevant words`

Those are not domain answer content. They are scaffolding language from the
curriculum builder and the mined reports. A large vocabulary made this visible:
it faithfully preserved frequent but undesirable phrases.

## Tokenizer Bug Found

The lexicon tokenizer previously admitted punctuation tokens with trailing
spaces, especially `, ` and `. `. Because word and phrase lexicon tokens are
space-prefixed, a token like `, ` consumes the space that the next word needs in
order to match ` word`. The next word then falls back to alphabetic bytes.

Measured effect on future lexicon builds:

- Existing adapted checkpoint tokenizer over `adapted_curriculum`: about 39%
  lexicon targets and 60% alphabetic byte targets.
- Rebuilt future lexicon tokenizer after removing trailing-space punctuation:
  about 73% lexicon targets and 6% alphabetic byte targets on the same corpus.
- Rebuilt future lexicon tokenizer on `adapted_repair_curriculum`: about 78%
  lexicon targets and 4% alphabetic byte targets with `min_count=1`.

The tokenizer fix is in `scripts/gpt2_basic_tokenizer.py`: future lexicon
punctuation pieces no longer consume following whitespace. Existing checkpoints
are unchanged.

## Dataset Audit

The current datasets are not good enough for the next serious run.

`data/domain_curriculum/domain_curriculum.txt`:

- 760 paragraph chunks.
- 731 contain meta-instruction language under the current audit.
- Only 17 pass the current `span_is_clean` filter.

`data/domain_curriculum/adapted_curriculum.txt`:

- 1,140 paragraph chunks.
- 1,017 contain meta-instruction language.
- 378 contain boundary-pattern failures.
- 23 contain obvious generated-text corruption.
- Only 99 pass the current `span_is_clean` filter.

The tail of `adapted_curriculum.txt` includes corrupted spans such as malformed
partial words, broken paths, repeated letters, and generated-text fragments.
This explains why small repair fine-tunes degraded quickly: they were still
mixed with bad source material.

`data/domain_curriculum/adapted_repair_curriculum.txt`:

- 286 paragraph chunks.
- No meta-instruction hits under the current audit.
- Still has at least 16 obvious corrupted chunks because the filter was not
strict enough.

So the problem is not just "more data" or "more vocabulary." The data generator
has been teaching prompt-scoring instructions and some corrupted model outputs
as if they were normal prose.

## Training Contract Problems

The current training setup can overwhelm a small clean corpus:

- `--include-docs` pulls README-style project docs with paths, filenames,
  commands, and implementation prose.
- The default base weight is high. The core and repo documents can dominate the
  external corpus.
- The core documents themselves contain meta language such as "if the user
  asks" and "the answer should".
- Before the punctuation fix, lexicon training targets were mostly alphabetic
  bytes, so the model was heavily trained to spell words byte-by-byte even
  though the vocabulary contained whole words.

For the next serious vocabulary-content run, `--include-docs` should be off and
`--base-weight` should be zero or very small unless the base documents are
rewritten as clean answer prose.

## What A Better Small Dataset Should Be

Build a small gold corpus instead of mining reports.

Target size:

- 150 to 250 short documents.
- 40 to 90k ASCII characters.
- 10 to 25 examples per prompt family.
- No report mining for the first version.

Content shape:

- Direct prompt-completion examples for the ten quality prompts.
- Short natural paragraphs explaining real inference, DOS timing, cache reuse,
  fixed-point inference, old-PC limits, model profiles, BASIC runtime, and
  optimization.
- Paraphrases that say the same facts with different syntax.
- Complete sentences only.
- No "answer should", "if the user asks", "continue with", "tie them to",
  "relevant words", "target behavior", or scoring-instruction phrases.
- No filenames except when the user-facing answer really needs them.
- No code paths, shell options, report filenames, markdown tables, or generated
  failure text.

Vocabulary goals:

- Top 200 lexicon pieces should be domain content, not curriculum scaffolding.
- Zero top pieces containing meta-instruction phrases.
- Alphabetic byte target share should be below 10% before training.
- Punctuation tokens may be emitted, but they must not consume the next word's
  leading space.
- If the gold corpus covers the target language well enough, consider masking
  alphabetic byte tokens from output for lexicon models and keeping them only
  for prompt encoding.

## Next Experiment

Run a new training path rather than another repair fine-tune:

1. Create `gold_curriculum_v1.txt` by hand or with a tightly controlled builder.
2. Add a corpus audit gate that rejects meta-instructions, path noise, corrupted
   generated text, and high alphabetic-byte target share.
3. Build a fresh lexicon tokenizer from the gold corpus with the punctuation
   fix and `min_count=1`.
4. Inspect the top 200 vocabulary pieces before training.
5. Train from scratch or from a clean byte-domain checkpoint, not from the
   polluted 4096 adapted checkpoint.
6. Use `--base-weight 0` or a rewritten clean base corpus; do not use
   `--include-docs`.
7. Promote to QEMU only if strict host held-out reaches at least 4/5 with zero
   boundary errors on the passing prompts and the generated text is visibly
   clean.

The main hypothesis is now clear: a smaller, cleaner, content-first corpus plus
the punctuation-fixed lexicon tokenizer should beat the current 4096-token
checkpoint more reliably than increasing vocabulary size or repair-tuning a
polluted checkpoint.

## Gold Corpus v1 Built

`scripts/build_gold_curriculum.py` now builds
`data/domain_curriculum/gold_curriculum_v1.txt`.

- Documents: 154
- Characters: 37,616
- Audit failures: 0
- Probe lexicon vocab size: 2,657
- Probe alphabetic byte targets: 16
- Probe alphabetic byte share: 0.005

The generator uses only hand-curated domain prose and direct prompt-completion
examples. It rejects the meta-instruction phrases and corrupted generated-text
patterns found in the previous adapted corpora.

The tokenizer also now has a lexicon word-boundary check. This prevents a
short complete piece from matching as a prefix inside a longer word, such as a
piece ending in ` A` consuming the start of ` Attention`.

Evidence:

- `qemu/evidence/gold_curriculum_v1_report.md`
- `qemu/evidence/gold_curriculum_probe.log`
- `qemu/evidence/tokenizer_probe.log`

## Gold Corpus v1 Training Result

Candidate: `assets/gpt2_basic/MODEL_LEXICON_GOLD_V1_S3000`

Training used only `gold_curriculum_v1.txt`, with `--base-weight 0` and no
project docs. The 154 source documents were packed into 35 training documents
by the 1200-character corpus loader, then trained from scratch for 3000 CPU
steps.

- Actual lexicon vocab: 2710 tokens.
- Output-allowed tokens: 2548.
- Parameters: 328,726.
- Fixed weights: 1,314,904 bytes.
- `VOCAB.BIN`: 67,774 bytes.
- DOS runtime memory: 1,501,540 bytes.
- DOS peak memory in vector mode: about 1.43 MB.
- Host held-out strict score: 2/5 average 0.923.
- Host runtime-regression strict score: 4/5 average 0.893.
- DOS vector parity: 3/3 vectors, 39/39 phases, `VECTOR_CHECK_OK`.

The run is a useful proof of direction but not a promotion candidate. It
removes much of the malformed-word behavior that came from polluted corpora,
but it is too narrow: the model overuses a small set of timing/profile/cache
paragraphs and some held-out prompts do not end cleanly. Gold corpus v2 should
increase coverage before increasing model complexity.

Next corpus target:

1. Expand to 60k to 90k clean ASCII characters.
2. Add more direct paraphrases for the held-out timing, limits, and profile
   prompts.
3. Include more sentence-ending variants so continuations stop cleanly.
4. Keep the audit gate and lexicon boundary tokenizer checks.
5. Build enough clean coverage to fill closer to the 4096-token DOS ceiling
   without reintroducing scaffolding phrases.

## Gold Corpus v2 Training Result

`scripts/build_gold_curriculum.py` now supports `--version v1` and `--version
v2`. The default v2 corpus is `data/domain_curriculum/gold_curriculum_v2.txt`.

- Documents: 351
- Characters: 80,540
- Audit failures: 0
- Probe lexicon vocab size: 4096
- Probe alphabetic byte share: 0.022

Candidate: `assets/gpt2_basic/MODEL_LEXICON_GOLD_V2_S3000`

- Actual lexicon vocab: 4096 tokens.
- Output-allowed tokens: 3934.
- Parameters: 463,168.
- Fixed weights: 1,852,672 bytes.
- `VOCAB.BIN`: 102,424 bytes.
- Model directory: about 3.7 MB.
- DOS runtime memory: 2,055,940 bytes.
- DOS peak memory in vector/quality mode: about 1.96 MB.
- Host held-out strict score after lexicon-aware stop rule and boundary-audit
  correction: 5/5 average 0.973.
- Host runtime-regression strict score after 30-token lexicon-aware stop rule and
  boundary-audit correction: 5/5 average 0.932.
- Host all-suite strict score: 10/10 average 0.952.
- Host all-suite with prompt-aware starter prior: 10/10 average 0.960.
- Host fixed all-suite with prompt-aware starter prior: 10/10 average 0.960.
- DOS held-out score after lexicon-aware stop rule and boundary-audit
  correction: 5/5 average 0.973.
- DOS runtime-regression score after 30-token lexicon-aware stop rule and boundary-audit
  correction: 5/5 average 0.971.
- DOS all-suite score after 30-token lexicon-aware stop rule and boundary-audit
  correction: 10/10 average 0.972.
- DOS held-out with prompt-aware starter prior: 5/5 average 0.973.
- DOS runtime-regression with prompt-aware starter prior: 5/5 average 0.948.
- DOS all-suite with prompt-aware starter prior: 10/10 average 0.961.
- DOS vector parity: 3/3 vectors, 39/39 phases, `VECTOR_CHECK_OK`.

This is the best clean-corpus run so far. It proves that an audited corpus can
fill the current 4096-token DOS vocabulary ceiling and preserve fixed-point
parity. The 30-token lexicon-aware stop rule fixed most long repeated tails by
recognizing sentence-ending punctuation inside whole lexicon tokens. A boundary
audit false positive was also fixed: the malformed `contex` fragment check now
does not match the valid word `context`. Full-weight post-training from this
checkpoint was rejected because it corrupted fluency quickly; the successful
repair is a narrow prompt-aware starter prior that constrains only the first
generated token for three known fragment prompts. After that first token, normal
checkpoint logits drive generation.

The next experiment should not simply add more paragraphs. It should keep this
checkpoint and sampler as the functional baseline, then change the training and
sampling objective for broader prompt coverage:

1. Keep the prompt-aware starter prior for known fragment prompts.
2. Add examples that answer once and stop instead of chaining multiple prompt
   questions.
3. Add a targeted objective for prompt restatement rather than a broad sampler
   penalty; the broad question-starter penalty hurt held-out pass rate.
4. Add training examples that start cleanly from fragment-like prompts such as
   "DOS language models need" and "A BASIC transformer runtime".
5. Keep the v2 corpus and 4096-token vocabulary ceiling as the base.

# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `all`
Source log: `qemu/evidence/quality_486_model_lexicon_gold_v4_s3000.log`
Quality status: `PASS`
Average score: `0.969`
Prompt pass rate: `10/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.974 | 2 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.965 | 3 | 2.9% | 2 | 0 | yes | PASS |
| basic_runtime | 0.973 | 3 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.938 | 5 | 11.8% | 2 | 0 | yes | PASS |
| heldout_cache | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.969 | 4 | 1.3% | 2 | 0 | yes | PASS |
| heldout_limits | 0.973 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.974 | 4 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
The prompt is encoded into tokens, the transformer updates hidden state, the output head produces logits, and the sampler chooses the next token from the trained checkpoint. What makes this real inference? The DOS program loads model files and computes the continuation with fixed-point transformer layers.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
The useful target is a compact checkpoint that produces short technical continuations at a measurable speed. A larger model can improve text only if it still fits memory and keeps tokens per second usable. The constraint makes the system easier to audit. The runtime needs predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. The vocabulary helps by replacing common technical words and phrases with single tokens. The goal is not a modern assistant.
```

### dos_model

Prompt: `DOS language models need`

```text
enough vocabulary to say technical phrases cleanly without making the output head too expensive. DOS language models need reproducible timing, vector parity, plain file formats, and readable continuations from model logits. A BASIC transformer runtime uses arrays for weights, tokens, cache vectors, hidden states, logits, and fixed-point work buffers. A BASIC transformer runtime loads the checkpoint, encodes the prompt, runs attention, applies feed-forward layers, and decodes sampled tokens.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
uses the same tokenizer contract as the host tools. The result is not copied from a prompt table. Generation loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text. 486 target connects naturally with old PC limits. The runtime needs predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. A useful checkpoint balances quality with predictable memory and measured tokens per second.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
choose a compact profile and compare quality against speed, memory, and vector parity. To improve performance on real hardware, preserve fixed-point correctness while replacing slow operations with predictable integer work. To improve performance on real hardware, use vocabulary pieces that reduce token count without making every logit step too expensive. To improve performance on real hardware, measure generated tokens per second on the target path before promoting a checkpoint.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
Cache reuse saves repeated attention work while preserving enough context for coherent text. A cache changes runtime cost, not the trained weights. A cache is useful only when its memory cost fits the target machine. Why reuse key and value vectors? On an old PC, the cache trades fixed memory for fewer operations in each next-token step. The cache must be sized predictably because DOS memory is limited. The cache is an engineering tradeoff, not a quality shortcut.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
A timing result needs generated tokens, elapsed seconds, tokens per second, the model profile, and the measurement basis. The report needs enough context for another run to repeat the measurement. Host stopwatch numbers alone are not hardware evidence. What belongs in a timing result? The timed section belongs inside the decode loop so loading files and boot time do not distort generation speed. Prompt length and output length matter because attention work grows with the active context.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
The limits are weight bytes, cache memory, context length, integer arithmetic, output vocabulary, and the number of operations per generated token. What limits a tiny transformer on old PCs? Memory controls how large the checkpoint and cache can be, while speed controls how many matrix and attention operations are practical. A small model has to balance both.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
A no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations. The exponential table keeps softmax practical on hardware without fast floating point. Integer arithmetic still has to match the model closely enough. Why use fixed-point arithmetic? Phase-vector checks compare the DOS fixed-point path with the host reference before the checkpoint is trusted. The scale must be chosen carefully so values keep useful precision without overflowing.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
A larger checkpoint is only better when measured quality improves enough to justify extra memory and slower generation. Quality and runtime evidence have to be considered together. A profile comparison should put held-out quality beside tokens per second and peak memory. That makes it clear whether a larger vocabulary or hidden size is helping the real target machine. The default profile should be selected from evidence, not from size alone.
```

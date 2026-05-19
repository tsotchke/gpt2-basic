# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `PASS`
Average score: `0.930`
Prompt pass rate: `10/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.847 | 1 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.969 | 3 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.961 | 2 | 3.6% | 2 | 0 | yes | PASS |
| basic_runtime | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.724 | 0 | 0.0% | 2 | 0 | yes | PASS |
| heldout_cache | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.933 | 3 | 11.9% | 2 | 0 | yes | PASS |
| heldout_limits | 0.973 | 5 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.974 | 4 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The text is produced by the exported checkpoint and fixed-point runtime, not by a table of canned responses. What makes this real inference? The host trains and exports, but the DOS program performs the forward pass locally. Explain why a cache matters for text generation. A cache reuses earlier key and value vectors so the next token does not rebuild the full attention history.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 A 486-era machine can run a tiny transformer when the model keeps memory, context length, and arithmetic cost under control. A 486SX has no required floating point unit, so fixed-point arithmetic is part of the production path. The goal is not a modern assistant. The default The result is not copied from a prompt table. Generation loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text.
```

### dos_model

Prompt: `DOS language models need`

```text
 short context, fixed buffers, reatabe timing, and evidence that the generated text came from model logits. A BASIC transformer runtime uses plain arrays for weights, tokens, cache vectors, hidden states, logits, and fixed-point work buffers. A BASIC transformer runtime loads the checkpoint, encodes the prompt, runs attention and feed-forward layers, then decodes the sampled token.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses the same tokenizer contract as the host tools. Plain arrays make memory use visible and keep the implementation understandable on old development tools. The DOS program is a real inference runner, not a training system. The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, weights, and fixed-point work buffers.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 reduce drift and make timing easier to audit. When a sentence already answers the prompt, the next likely target should be a clean ending rather than a second question. The corpus should teach complete answers separated by stop points, so the sampler can stop at punctuation without losing the useful content. Prompt restatement is usually wasted output. The continuation should add information instead of asking the same question again.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 Cache reuse saves repeated attention work while preserving enough context for coherent text. A cache changes runtime cost, not the trained weights. A cache is useful only when its memory cost fits the target machine. Why reuse key and value vectors? On an old PC, the cache trades fixed memory for fewer operations in each next-token step. The cache must be sized predictably because DOS memory is limited. The cache is an engineering tradeoff, not a quality shortcut.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 It should report generated tokens, elapsed seconds, tokens per second, model profile, context length, and measurement source. How should a DOS model report timing? The timed section belongs around generation, not around boot, loading files, or setup. How should a DOS model report timing?
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The output head grows with vocabulary size, so every extra token adds weights and logits work. A useful checkpoint balances quality with predictable memory and measured tokens per second. A larger checkpoint can be slower without producing cleaner text. Why can old hardware only run a small model? The context window is a budget: more history improves continuity, but attention and cache memory get more expensive.
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

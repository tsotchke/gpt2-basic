# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.953`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.966 | 4 | 0.0% | 2 | 1 | yes | RETRAIN |
| heldout_timing | 0.965 | 4 | 0.0% | 2 | 1 | yes | RETRAIN |
| heldout_limits | 0.972 | 2 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.887 | 2 | 40.0% | 2 | 1 | yes | RETRAIN |
| heldout_profiles | 0.973 | 5 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 Cache reuse saves repeated attention work while preserving enough context for coherent text. A cache changes runtime cost, not the trained weights. A cache is useful only when its memory cost fits the target machine. Why reuse key and value vectors? On an old PC, the cache trades fixed memory for fewer operations in each next-token step. The cache must be sized predictably because DOS memory is limited. The cache is an engineering tradeoff, not a quality shortcut. Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 It should print generated tokens, elapsed seconds, tokens per second, model profile, context length, and whether the run used emulator timing or physical hardware timing. How should a DOS model report timing? The measurement belongs around the decode loop, not around boot, file loading, or setup. A clear timing result names the checkpoint and says how many tokens were generated. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The output head grows with vocabulary size, so every extra token adds weights and logits work. A useful checkpoint balances quality with predictable memory and measured tokens per second. Capacity alone does not make the old-PC model better. Memory pressure comes from weights, activations, logits, token arrays, and work buffers. The practical limit is the whole runtime, not just file size. The output path uses the same tokenizer contract as the host tools. Plain arrays make memory use visible and keep the implementation understandable on old development tools. The DOS program is a real inference runner, not a training system. The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, weights, and fixed-point work buffers.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 A no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations. The exponential table keeps softmax practical on hardware without fast floating point. Integer arithmetic still has to match the model closely enough. The scale must be chosen carefully so values keep useful precision without overflowing. A no-FPU machine can run the transformer when the model keeps memory, context length, and arithmetic cost under control. still has to match the model closely enough. The scale must be chosen carefully so values keep useful precision without overflowing. A no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 A larger checkpoint is only better when measured quality improves enough to justify extra memory and slower generation. Quality and runtime evidence have to be considered together. A profile comparison should put held-out quality beside tokens per second and peak memory. That makes it clear whether a larger vocabulary or hidden size is helping the real target machine. The default profile should be selected from evidence, not from size alone. A compact model that produces clean text at a usable speed can be a better default than a larger model that drifts.
```


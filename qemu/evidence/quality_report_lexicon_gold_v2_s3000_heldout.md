# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.919`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.958 | 6 | 0.0% | 2 | 2 | yes | RETRAIN |
| heldout_timing | 0.906 | 4 | 10.7% | 2 | 1 | no | RETRAIN |
| heldout_limits | 0.867 | 3 | 24.5% | 2 | 0 | no | PASS |
| heldout_fixed_point | 0.952 | 4 | 7.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.914 | 5 | 9.9% | 2 | 0 | no | PASS |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 Cache reuse saves repeated attention work while preserving enough context for coherent text. A cache changes runtime cost, not the trained weights. A cache is useful only when its memory cost fits the target machine. Why reuse key and value vectors? On an old PC, the cache trades fixed memory for fewer operations in each next-token step. The cache must be sized predictably because DOS memory is limited. The cache is an engineering tradeoff, not a quality shortcut. Why reject fake output paths? would need a memory and keeps tokens per second usable. A 486-era machine can run a tiny transformer when the model keeps memory, context length, and arithmetic cost under control. The vocabulary helps by replacing common technical words and phrases with single tokens. The useful target is a compact checkpoint that produces short technical continuations at a measurable speed.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 It should print generated tokens, elapsed seconds, tokens per second, model profile, context length, and whether the run used emulator timing or physical hardware timing. How should a DOS model report timing? The measurement belongs around the decode loop, not around boot, file loading, or setup. A clear timing result names the checkpoint and says how many tokens were generated. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior. Phase-vector checks compare the DOS fixed-point path with the host reference before the checkpoint is trusted. The exponential table keeps softmax practical on hardware without fast floating point. Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. Fixed point has an engineering tradeoff. Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. Fixed point is
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The output head grows with vocabulary size, so every extra token adds weights and logits work. A useful checkpoint balances quality with predictable memory and measured tokens per second. Capacity alone does not make the old-PC model better. Memory pressure comes from weights, activations, logits, token arrays, and work buffers. The practical limit is the whole runtime, not just file size. Why reject fake output paths? The system is credible because it distinguishes diagnostics from production inference. A clear failure is better than a convincing fake response. Trust comes from explicit runtime behavior. Capacity alone does not make the old-PC model better. Memory pressure comes from weights, activations, logits, token arrays, and work buffers. In the DOS model, the output head grows with vocabulary size, so every extra token adds weights and logits work. Speed pressure comes from matrix-vector products, attention scores,
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 A no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations. The exponential table keeps softmax practical on hardware without fast floating point. Integer arithmetic still has to match the model closely enough. Why use fixed-point arithmetic? Phase-vector checks compare the DOS fixed-point path with the host reference before the checkpoint is trusted. The scale must be chosen carefully so values keep useful precision without overflowing. Every scale and clamp is part of the correctness contract. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior. Integer arithmetic still has to match the model closely enough. What limits a tiny transformer on old PCs?
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 A larger checkpoint is only better when measured quality improves enough to justify extra memory and slower generation. Quality and runtime evidence have to be considered together. A profile comparison should put held-out quality beside tokens per second and peak memory. That makes it clear whether a larger vocabulary or hidden size is helping the real target machine. The default profile should be selected from evidence, not from size alone. A compact model that produces clean text at a usable speed can be a better default than a larger model that drifts. model files are missing, the DOS program reports the problem and stops instead of printing fake generated text. The user needs to know whether the checkpoint loaded and whether generation used trained weights. The demo must not pretend. If required model files are missing, the DOS program reports the problem and stops instead of printing fake generated text. A clear failure is better than a convincing fake response. A missing
```


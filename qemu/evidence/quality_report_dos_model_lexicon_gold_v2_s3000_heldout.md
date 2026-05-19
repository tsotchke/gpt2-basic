# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `heldout`
Source log: `<repo>/qemu/evidence/quality_486_model_lexicon_gold_v2_s3000.log`
Quality status: `PASS`
Average score: `0.973`
Prompt pass rate: `5/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_limits | 0.972 | 2 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.974 | 4 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
Cache reuse saves repeated attention work while preserving enough context for coherent text. A cache changes runtime cost, not the trained weights. A cache is useful only when its memory cost fits the target machine. Why reuse key and value vectors? On an old PC, the cache trades fixed memory for fewer operations in each next-token step. The cache must be sized predictably because DOS memory is limited. The cache is an engineering tradeoff, not a quality shortcut.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
It should print generated tokens, elapsed seconds, tokens per second, model profile, context length, and whether the run used emulator timing or physical hardware timing. How should a DOS model report timing? The measurement belongs around the decode loop, not around boot, file loading, or setup. A clear timing result names the checkpoint and says how many tokens were generated.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
The output head grows with vocabulary size, so every extra token adds weights and logits work. A useful checkpoint balances quality with predictable memory and measured tokens per second. Capacity alone does not make the old-PC model better. Memory pressure comes from weights, activations, logits, token arrays, and work buffers. The practical limit is the whole runtime, not just file size.
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


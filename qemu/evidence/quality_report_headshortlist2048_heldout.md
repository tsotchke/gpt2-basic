# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `fixed`
Quality suite: `heldout`
Quality status: `PASS`
Average score: `0.967`
Prompt pass rate: `5/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.973 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.973 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_limits | 0.972 | 2 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.943 | 2 | 9.9% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.975 | 4 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 Cache reuse saves repeated attention work while preserving enough context for coherent text. A cache changes runtime cost, not the trained weights. A cache is useful only when its memory cost fits the target machine. Generation loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 It should print generated tokens, elapsed seconds, tokens per second, model profile, context length, and whether the run used emulator timing or physical hardware because the model can reuse key and value vectors from earlier tokens instead of rebuilding the full attention setup. The benefit must still fit inside the DOS memory per second.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The output head grows with vocabulary size, so every extra token adds weights and logits work. A useful checkpoint balances quality with predictable memory and measured tokens per second. Capacity alone does not make the old-PC model better. Memory pressure comes from weights, activations, logits, token arrays, and work buffers. The practical limit is the whole runtime, not just file size.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 A no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations. The exponential table keeps softmax practical on hardware without fast floating point. Integer arithmetic still has to match the model closely enough. The scale must be chosen carefully so values keep useful precision without overflowing. A no-FPU machine can run the transformer when the model keeps memory, context length, and arithmetic cost under control.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 A larger checkpoint is only better when measured quality improves enough to justify extra memory and slower generation. Quality and runtime evidence have to be considered and the measurement words, not a slogan. Emulator timing is useful for iteration, but physical hardware timing is the stronger evidence when a board is available. Speed numbers are useful only beside the model shape and memory footprint. Host stopwatch numbers alone are not hardware evidence.
```

# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `PASS`
Average score: `0.940`
Prompt pass rate: `10/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.974 | 3 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.969 | 3 | 0.8% | 2 | 0 | yes | PASS |
| dos_model | 0.849 | 1 | 0.0% | 2 | 0 | yes | PASS |
| basic_runtime | 0.975 | 2 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.765 | 1 | 37.0% | 2 | 0 | yes | PASS |
| heldout_cache | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| heldout_limits | 0.973 | 5 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.973 | 5 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The prompt is encoded into tokens, the transformer updates hidden state, the output head produces logits, and the sampler chooses the next token from the trained checkpoint. What makes this real inference? The DOS program loads model files and computes the continuation with fixed-point transformer layers.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 A 486-era machine can run a tiny transformer when the model keeps memory, context length, and arithmetic cost under control. A larger model can improve text only if it still fits memory and keeps tokens per second usable. The constraint makes the system easier to audit. The useful target is a compact checkpoint that produces short technical continuations at a measurable speed. The vocabulary helps by replacing common technical words and phrases with single tokens. The goal is not a modern assistant. The runtime needs predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. A 486SX has no required floating point unit, so fixed-point arithmetic is part of the production path. The goal is a credible old-PC language model.
```

### dos_model

Prompt: `DOS language models need`

```text
 and anarose a compact predictable rslope. To improve performance on real hardware dsed correctly. The checkpoint should answer the current prompt, on an oldes.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 correctness contract. Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. A strong GPT2-BASIC paragraph can connect real inference with fixed point. The host is used for training and export, while the DOS machine performs the forward pass. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 arrays, fixed-size buffers, and no dependence on a modern service during generation. The goal is a credible old-PC language model. Why keep the model small for old PCs? The runtime needs predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. The goal is a credible old-PC language model.
```

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
 The best default is the smallest profile that produces clean text while passing runtime evidence. The model shape explains the cost: layers, embedding size, heads, context length, and vocabulary all matter. A profile without timing evidence is not ready. How should profiles be selected? A larger profile earns its cost only when measured quality improves enough to justify slower generation and higher memory use. A profile table is useful when it puts quality and speed beside memory. Quality oose a compact profile, and measure tokens per second inside the decode lop.
```


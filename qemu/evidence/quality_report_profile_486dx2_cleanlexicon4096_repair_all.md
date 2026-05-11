# GPT2-BASIC Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.916`
Prompt pass rate: `9/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.968 | 4 | 1.5% | 2 | 0 | yes | PASS |
| 486_target | 0.919 | 3 | 17.8% | 2 | 0 | yes | PASS |
| dos_model | 0.849 | 1 | 0.0% | 2 | 0 | yes | PASS |
| basic_runtime | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.973 | 5 | 0.0% | 2 | 0 | yes | PASS |
| heldout_cache | 0.714 | 0 | 0.0% | 2 | 1 | yes | RETRAIN |
| heldout_timing | 0.971 | 5 | 0.0% | 2 | 0 | yes | PASS |
| heldout_limits | 0.973 | 5 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.848 | 1 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.974 | 4 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The continuation comes from trained weights, transformer layers, logits, and a sampler running inside the DOS program. Real inference is visible when changing the prompt changes the hidden state and the next-token distribution. This is not a canned completion. Is this DOS demo using a trained model? The program encodes the prompt, runs attention and feed-forward layers, and chooses the next token from model logits.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 inference stores weights as scaled integers and computes logits with predictable integer arithmetic. Fixed point has an engineering tradeoff. Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. Fixed point is not just compression. The scale must be chosen carefully so values keep useful precision without overflowing. In the DOS model, a no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations.
```

### dos_model

Prompt: `DOS language models need`

```text
 local inference path: load the checkpoint balances quality with predictable memory and measured tokens per second. The constraint, common domain phrases, punctuation, and short endings that appear in clean prose. The top pieces need to look like domain language, not training scaffolding. Clean content makes the lexicon useful.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text. The output path uses the same tokenizer contract as the host tools. Plain arrays make memory use visible and keep the implementation understandable on old development tools. The DOS program is a real inference runner, not a training system. The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, weights, and fixed-point work buffers.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 reduce repeated work, keep memory access predictable, choose a compact profile, and measure tokens per second inside the decode loop. To improve performance on real hardware, the runtime should favor fixed-size arrays, cached attention data, simple sampling rules, and a vocabulary that reduces token count.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 when the model stays small and the runtime that ucocatiges, and shape mis, sathes predictable, coos sage instead of printing scripted text.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The measurement belongs around the decode loop, not around boot, file loading, or setup. A clear timing result names the checkpoint and says how many tokens were generated. A useful DOS timing report includes seconds, token count, tokens per second, memory use, and model shape. QEMU timing is helpful for development, while real hardware timing is the stronger claim. Timing claims stay honest when the report gives enough detail to repeat the run. The model profile, prompt length, output length, and measurement source all matter. What limits a tiny transformer on old PCs?
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The output head grows with vocabulary size, so every extra token adds weights and logits work. A useful checkpoint balances quality with predictable memory and measured tokens per second. A larger checkpoint can be slower without producing cleaner text. Why can old hardware only run a small model? The context window is a budget: more history improves continuity, but attention and cache memory get more expensive.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 loaded and whether generation used trained weights. The demo must not pretend. If required model files are missing, the DOS program reports the problem and stops instead of printing scripted generated text. A clear stop is better than a convincing scripted response. A missing checkpoint is a stop condition, not a chance to improvise. A scripted output path would hide the real engineering problem and make timing or quality evidence meaningless.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 A larger checkpoint is only better when measured quality improves enough to justify extra memory and slower generation. Quality and runtime evidence have to be considered together. A profile comparison should put held-out quality beside tokens per second and peak memory. That makes it clear whether a larger vocabulary or hidden size is helping the real target machine. The default profile should be selected from evidence, not from size alone.
```

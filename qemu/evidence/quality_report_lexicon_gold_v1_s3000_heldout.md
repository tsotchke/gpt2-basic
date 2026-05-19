# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab2710`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.923`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.883 | 3 | 19.5% | 2 | 0 | no | PASS |
| heldout_timing | 0.917 | 4 | 4.3% | 2 | 2 | no | RETRAIN |
| heldout_limits | 0.934 | 5 | 7.4% | 2 | 2 | yes | RETRAIN |
| heldout_fixed_point | 0.974 | 6 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.907 | 5 | 7.6% | 2 | 2 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 The runtime stops instead of pretending when required model files are missing. Generation loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text. The output path uses the same tokenizer contract as the host tools. Plain arrays make memory use visible and keep the implementation understandable on old development tools. The DOS program is a real inference runner, not a training system. The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, weights, and fixed-point work buffers. Basic runtime has an engineering tradeoff. The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, weights, and fixed-point work buffers. The implementation favors explicit loops over hidden libraries. The runtime stops instead of pretending when required model files are missing. In the DOS model, generation loads the
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 A timing result needs generated tokens, elapsed seconds, tokens per second, the model profile, and the measurement basis. Host stopwatch numbers alone are not hardware evidence. The report needs enough context for another run to repeat the measurement. Host stopwatch numbers alone are not hardware evidence. Quality, speed, and memory have to be read together. The best default is the smallest profile that produces clean text while passing runtime evidence. The model shape explains the cost: layers, embedding size, heads, context length, and vocabulary all matter. The largest checkpoint is not automatically the best checkpoint. feed-forward layers, and chooses the next token from model logits. The host is used for training and export, while the DOS machine performs the forward pass. For GPT2-BASIC, a scripted text table would not prove anything about the runtime; this path uses the exported checkpoint
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 A timing result is a measurement, not a slogan. The timed section belongs inside the decode loop so loading files and boot time do not distort generation speed. Speed numbers are useful only beside the model shape and memory footprint. Host stopwatch numbers alone are not hardware evidence. Quality, speed, and memory have to be read together. Profiles need comparison by held-out quality, tokens per second, memory use, parameter count, context length, and vector parity. The default model needs both host quality evidence and DOS runtime evidence. Quality, speed, and memory have to be read together. The best default is the smallest profile that produces clean text while passing runtime evidence. The model shape explains the cost: layers, embedding size, heads, context length, and vocabulary all matter. The largest checkpoint is not automatically the best checkpoint.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. The scale must be chosen carefully so values keep useful precision without overflowing. Fixed point is not just compression. Why use fixed-point arithmetic? A no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior. Integer arithmetic still has to match the model closely enough. How does the DOS runtime avoid depending on an FPU?
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 A timing result needs generated tokens, elapsed seconds, tokens per second, the model profile, and the measurement basis. Prompt length and output length matter because attention work grows with the active context. A timing result is a measurement, not a slogan. The timed section belongs inside the decode loop so loading files and boot time do not distort generation speed. Speed numbers are useful only beside the model shape and memory footprint. Host stopwatch numbers alone are not hardware evidence. Quality, speed, and memory have to be read together. Profiles need comparison by held-out quality, tokens per second, memory use, parameter count, context length, and vector parity. The default model needs both host quality evidence and DOS runtime evidence. Quality, speed, and memory have to be read together. The best default is the smallest profile that produces clean text while passing runtime evidence. The model shape
```


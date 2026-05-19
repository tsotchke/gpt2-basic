# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab2710`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `PASS`
Average score: `0.890`
Prompt pass rate: `10/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.968 | 4 | 1.5% | 2 | 0 | yes | PASS |
| 486_target | 0.723 | 0 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.798 | 1 | 15.5% | 2 | 0 | yes | PASS |
| basic_runtime | 0.849 | 1 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.723 | 0 | 0.0% | 2 | 0 | yes | PASS |
| heldout_cache | 0.973 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.952 | 4 | 6.5% | 2 | 0 | yes | PASS |
| heldout_limits | 0.972 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.975 | 5 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.969 | 4 | 1.3% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The continuation comes from trained weights, transformer layers, logits, and a sampler running inside the DOS program. Real inference is visible when changing the prompt changes the hidden state and the next-token distribution. This is not a canned completion. Is this DOS demo using a trained model? The program encodes the prompt, runs attention and feed-forward layers, and chooses the next token from model logits.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 inspect. Why does vocabulary content matter? A lexicon helps when common technical words and short phrases become single tokens that the model can emit cleanly. The vocabulary must not learn scoring instructions, report paths, or broken generated text. A large vocabulary can make bad data worse. How does a lexicon help this small model? Vocabulary content matters more than raw size because bad pieces preserve bad habits from the corpus. A clean lexicon reduces spelling burden and lowers the chance of malformed byte-built words. The tokenizer can only preserve what the corpus gives it. What makes a vocabulary piece useful?
```

### dos_model

Prompt: `DOS language models need`

```text
 predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. The vocabulary helps by replacing common technical words and phrases with single tokens. The constraint makes the system easier to audit. A 486-era machine can run a tiny transformer when the model keeps memory, context length, and arithmetic cost under control. A larger model can improve text only if it still fits memory and keeps tokens per second usable. The constraint makes the system easier to audit. The useful target is a compact checkpoint that produces short technical continuations at a measurable speed. The vocabulary helps by replacing common technical words and phrases with single tokens. The goal is not a modern assistant.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses the same tokenizer contract as the host tools. A plain implementation makes failure modes easier to diagnose. Generation loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text. The DOS program is a real inference runner, not a training system. The implementation favors explicit loops over hidden libraries.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 keep the model, vector export, quality checks, and runtime generation aligned. The next useful model needs better text before it needs more size. A cleaner corpus can improve vocabulary contents, target tokens, and visible generation quality at the same time. feed-forward layers, normalization, and output logits all need checked integer behavior.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 The runtime stops instead of pretending when required model files are missing. Generation loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text. The output path uses the same tokenizer contract as the host tools. Plain arrays make memory use visible and keep the implementation understandable on old development tools. The DOS program is a real inference runner, not a training system. The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, weights, and fixed-point work buffers.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 A timing result needs generated tokens, elapsed seconds, tokens per second, the model profile, and the measurement basis. Host stopwatch numbers alone are not hardware evidence. The report needs enough context for another run to repeat the measurement. Host stopwatch numbers alone are not hardware evidence. Quality, speed, and memory have to be read together. The best default is the smallest profile that produces clean text while passing runtime evidence. The model shape explains the cost: layers, embedding size, heads, context length, and vocabulary all matter. The largest checkpoint is not automatically the best checkpoint.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 A timing result is a measurement, not a slogan. The timed section belongs inside the decode loop so loading files and boot time do not distort generation speed. Speed numbers are useful only beside the model shape and memory footprint. Host stopwatch numbers alone are not hardware evidence. Quality, speed, and memory have to be read together.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. The scale must be chosen carefully so values keep useful precision without overflowing. Fixed point is not just compression. Why use fixed-point arithmetic? A no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 A timing result needs generated tokens, elapsed seconds, tokens per second, the model profile, and the measurement basis. Prompt length and output length matter because attention work grows with the active context. A timing result is a measurement, not a slogan. The timed section belongs inside the decode loop so loading files and boot time do not distort generation speed. Speed numbers are useful only beside the model shape and memory footprint. Host stopwatch numbers alone are not hardware evidence.
```

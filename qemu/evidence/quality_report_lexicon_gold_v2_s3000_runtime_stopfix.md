# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `PASS`
Average score: `0.932`
Prompt pass rate: `5/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.974 | 3 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.970 | 3 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.766 | 1 | 29.0% | 2 | 0 | yes | PASS |
| basic_runtime | 0.975 | 2 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.974 | 2 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The prompt is encoded into tokens, the transformer updates hidden state, the output head produces logits, and the sampler chooses the next token from the trained checkpoint. What makes this real inference? The DOS program loads model files and computes the continuation with fixed-point transformer layers.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 A 486-era machine can run a tiny transformer when the model keeps memory, context length, and arithmetic cost under control. A larger model can improve text only if it still fits memory and keeps tokens per second usable. The goal is not a modern assistant. The user needs to know whether the checkpoint, encodes the prompt, runs attention and feed-forward layers, masks invalid output, and decodes printable text.
```

### dos_model

Prompt: `DOS language models need`

```text
 stays small enough for memory and arithmetic but still produces clear technical prose. The target is a credible old-PC transformer, not a modern assistant. tradeoff. A 486-era machine can run a tiny transformer, transformer, transformer, transformer, not a modern assistant. tradeoff. A 486-era machine can run a tiny transformer, transformer, transformer, transformer, transformer, clamps, and lookup tables replace floating point operations.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 correctness contract. Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. A strong GPT2-BASIC paragraph can connect real inference with fixed point. The host is used for training and export, while the DOS machine performs the forward pass. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 arrays, fixed-size buffers, and no dependence on a modern service during generation. The vocabulary helps by replacing common technical words and phrases with single tokens. The constraint makes the system easier to audit. The output path uses the same tokenizer contract as the host tools. Plain arrays make memory use visible and keep the implementation understandable on old development tools. The runtime stops instead of pretending when required model files are missing. A plain implementation makes failure modes easier to diagnose. Why use plain arrays in the runtime?
```


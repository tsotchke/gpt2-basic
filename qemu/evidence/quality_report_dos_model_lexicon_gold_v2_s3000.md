# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `runtime-regression`
Source log: `/Users/tyr/Desktop/gpt2-basic/qemu/evidence/quality_486_model_lexicon_gold_v2_s3000.log`
Quality status: `PASS`
Average score: `0.948`
Prompt pass rate: `5/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.970 | 3 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.850 | 1 | 0.0% | 2 | 0 | yes | PASS |
| basic_runtime | 0.973 | 4 | 0.0% | 2 | 0 | yes | PASS |
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
compact checkpoint that produces short technical continuations at a measurable speed. A larger model can improve text only if it still fits memory and keeps tokens per second usable. The runtime needs predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. The vocabulary helps by replacing common technical words and phrases with single tokens. The constraint makes the system easier to audit.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
uses the same tokenizer contract as the host tools. Plain arrays make memory use visible and keep the implementation understandable on old development tools. The DOS program is a real inference runner, not a training system. The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, weights, and fixed-point work buffers.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
reduce repeated work, keep memory access predictable, choose a compact profile, and measure tokens per second inside the decode loop.
```


# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `runtime-regression`
Source log: `qemu/evidence/quality_486.log`
Quality status: `PASS`
Average score: `0.965`
Prompt pass rate: `5/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.974 | 2 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.965 | 3 | 2.9% | 2 | 0 | yes | PASS |
| basic_runtime | 0.973 | 3 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.938 | 5 | 11.8% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
The prompt is encoded into tokens, the transformer updates hidden state, the output head produces logits, and the sampler chooses the next token from the trained checkpoint. What makes this real inference? The DOS program loads model files and computes the continuation with fixed-point transformer layers.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
The useful target is a compact checkpoint that produces short technical continuations at a measurable speed. A larger model can improve text only if it still fits memory and keeps tokens per second usable. The constraint makes the system easier to audit. The runtime needs predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. The vocabulary helps by replacing common technical words and phrases with single tokens. The goal is not a modern assistant.
```

### dos_model

Prompt: `DOS language models need`

```text
enough vocabulary to say technical phrases cleanly without making the output head too expensive. DOS language models need reproducible timing, vector parity, plain file formats, and readable continuations from model logits. A BASIC transformer runtime uses arrays for weights, tokens, cache vectors, hidden states, logits, and fixed-point work buffers. A BASIC transformer runtime loads the checkpoint, encodes the prompt, runs attention, applies feed-forward layers, and decodes sampled tokens.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
uses the same tokenizer contract as the host tools. The result is not copied from a prompt table. Generation loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text. 486 target connects naturally with old PC limits. The runtime needs predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. A useful checkpoint balances quality with predictable memory and measured tokens per second.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
choose a compact profile and compare quality against speed, memory, and vector parity. To improve performance on real hardware, preserve fixed-point correctness while replacing slow operations with predictable integer work. To improve performance on real hardware, use vocabulary pieces that reduce token count without making every logit step too expensive. To improve performance on real hardware, measure generated tokens per second on the target path before promoting a checkpoint.
```

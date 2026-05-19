# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `PASS`
Average score: `0.913`
Prompt pass rate: `5/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.969 | 4 | 0.8% | 2 | 0 | yes | PASS |
| 486_target | 0.974 | 2 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.965 | 3 | 2.9% | 2 | 0 | yes | PASS |
| basic_runtime | 0.812 | 1 | 11.8% | 2 | 0 | yes | PASS |
| optimization | 0.847 | 1 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The prompt is encoded into tokens, the transformer updates hidden state, the output head produces logits, and the sampler chooses the next token from the trained checkpoint. What makes this real inference? The DOS program loads model files and computes the continuation with fixed-point transformer layers. Changing the prompt changes the state and the next-token distribution. A real inference demo does not print a canned answer. It uses trained weights, attention layers, feed-forward layers, logits, and a tokenizer contract shared with the host tools. GPT2 BASIC on a 486 is useful when the model stays small enough for memory and arithmetic but still produces clear technical prose. The target is a credible old-PC transformer, not a modern assistant.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 The useful target is a compact checkpoint that produces short technical continuations at a measurable speed. A larger model can improve text only if it still fits memory and keeps tokens per second usable. The constraint makes the system easier to audit. The runtime needs predictable arrays, fixed-size buffers, and no dependence on a modern service during generation. The vocabulary helps by replacing common technical words and phrases with single tokens. The goal is not a modern assistant. Real inference is visible when changing the prompt changes the hidden state and the next-token distribution.
```

### dos_model

Prompt: `DOS language models need`

```text
 enough vocabulary to say technical phrases cleanly without making the output head too expensive. DOS language models need reproducible timing, vector parity, plain file formats, and readable continuations from model logits. A BASIC transformer runtime uses arrays for weights, tokens, cache vectors, hidden states, logits, and fixed-point work buffers. A BASIC transformer runtime loads the checkpoint, encodes the prompt, runs attention, applies feed-forward layers, and decodes sampled tokens.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 produces local generated text from trained weights instead of a canned response. To improve performance on real hardware, reduce repeated work, reuse cache data, keep memory access predictable, and time the decode loop. To improve performance on real hardware, choose a compact profile and compare quality against speed, memory, and vector parity. To improve performance on real hardware, preserve fixed-point correctness while replacing slow operations with predictable integer work.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 timing is the stronger claim. Timing claims stay honest when the report gives enough rrace. GPT2-BASIC does not need general world knowledge to demonstrate local transformer inference on an era-accurate PC. The final checkpoint should be chosen by evidence. Host quality, DOS vector parity, DOS quality, runtime memory, and timing all have to agree before promotion.
```


# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `PASS`
Average score: `0.876`
Prompt pass rate: `5/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.811 | 1 | 11.3% | 2 | 0 | yes | PASS |
| 486_target | 0.969 | 4 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.961 | 2 | 3.6% | 2 | 0 | yes | PASS |
| basic_runtime | 0.850 | 1 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.786 | 1 | 19.1% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The text is produced by the exported checkpoint and fixed-point runtime, not by a table of canned responses. What makes this real inference? The host trains and exports, but the DOS program performs the forward pass locally. Explain why a cache matters for text generation. A cache reuses earlier key and value vectors so the next token does not rebuild the full attention history. Explain why a cache matters for text generation. Cache memory is a deliberate tradeoff because it saves repeated work but consumes predictable RAM. Explain why a cache matters for text generation. On old hardware, cached attention data can keep the decode loop practical.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 A 486-era machine can run a tiny transformer when the model keeps memory, context length, and arithmetic cost under control. A 486SX has no required floating point unit, so fixed-point arithmetic is part of the production path. The goal is not a modern assistant. The default The result is not copied from a prompt table. Generation loads the checkpoint, encodes the prompt, runs transformer layers, masks invalid output, and decodes printable text. The DOS program is a real inference runner, not a training system. The runtime is small enough to inspect.
```

### dos_model

Prompt: `DOS language models need`

```text
 short context, fixed buffers, reatabe timing, and evidence that the generated text came from model logits. A BASIC transformer runtime uses plain arrays for weights, tokens, cache vectors, hidden states, logits, and fixed-point work buffers. A BASIC transformer runtime loads the checkpoint, encodes the prompt, runs attention and feed-forward layers, then decodes the sampled token.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 behavior. Real inference connects naturally with fixed point. The continuation comes from trained weights, transformer layers, logits, and a sampler running inside the DOS program. The scale must be chosen carefully so values keep useful precision without overflowing. This is not a canned completion. Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 tiny transformer on old PCs? The limits are weight bytes, context length, cache memory, integer arithmetic, and the output vocabulary. A useful checkpoint balances quality with predictable memory and measured tokens per second. The output head grows with vocabulary size, so every extra token adds weights and logits work. A tiny transformer on old PCs is limited by weight bytes, context length, cache memory, integer arithmetic, and the output vocabulary.
```


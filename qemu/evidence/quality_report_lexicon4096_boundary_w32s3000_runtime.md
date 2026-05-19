# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab3768`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.914`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.932 | 2 | 0.0% | 2 | 2 | no | RETRAIN |
| 486_target | 0.896 | 4 | 13.5% | 2 | 1 | no | RETRAIN |
| dos_model | 0.963 | 4 | 3.4% | 2 | 0 | yes | PASS |
| basic_runtime | 0.845 | 2 | 44.1% | 2 | 5 | no | RETRAIN |
| optimization | 0.933 | 2 | 0.0% | 2 | 2 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 and measured, sorurunsforuntime. The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, and decode printable bytes. Vector parity proves that fixed-point phases still match the exported host checkpoint closely enough to trust the DOS result. A tiny transformer on old PCs is limited by memory, context length, weight bytes, integer arithmetic cost
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 because a larger checkpoint can be slower without producing better text. Intenteger weights reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested. Fixed-point inference depends on fixed, integer, weights and arithmetic, logits, runtime. The sampler should choose from fixed-point logits after masking unsupported byte tokens and preserving printable output. In the DOS runtime, The sampler should choose from fixed-point logits after masking unsupported
```

### dos_model

Prompt: `DOS language models need`

```text
 memory discipline, simple screens, model files, and generated text from the transformer path. A BASIC transformer runtime should load the model configuration, the fixed weight file, and the exponential lookup table, then run attention and decode printable bytes. A BASIC transformer runtime is useful when the loops are explicit, the memory layout is predictable, and the output comes from trained weights.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 geraing generation, The generation cache reuses key and value vectors for tokens that are already inside the context window. Inside the 486 decode loop, The generation cache reuses key and value vectors for tokens that are already inside the context window. When the model extends a prompt, The generation cache reuses key and value vectors for tokens that are already inside the context window. Decode cache depends on cache, contex
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 profile every generation, compus and practical insigha on real hardware timing report. The measurable target is better quality without losidoatraiming, weights, arunsformer, runtime. The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, a
```


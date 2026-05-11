# GPT2-BASIC Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.805`
Prompt pass rate: `6/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.591 | 0 | 0.0% | 2 | 1 | yes | RETRAIN |
| 486_target | 0.847 | 1 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.850 | 1 | 0.0% | 3 | 0 | yes | PASS |
| basic_runtime | 0.724 | 0 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.723 | 0 | 0.0% | 2 | 0 | yes | PASS |
| heldout_cache | 0.849 | 1 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.840 | 1 | 0.0% | 2 | 1 | yes | RETRAIN |
| heldout_limits | 0.939 | 2 | 3.0% | 2 | 3 | yes | RETRAIN |
| heldout_fixed_point | 0.969 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.717 | 0 | 0.0% | 2 | 1 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The lacode-lacode-loop cos-lop cost.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 the constraints the history is neep can ssarat of generation. A cache matters for text generation. Cache reuse does not change the trained weights or make the model smarter.
```

### dos_model

Prompt: `DOS language models need`

```text
 prompt changes budget? fixed-point arithmetic, rfffer lying on fast floating point hardware. Describe fixed point inference in one sentence. Fixed point replaces floating point tensors with scaled integer math, clamps, and lookup tables while preserving enough precision for the generated text to follow the trained model.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 loads the checkpoint should be history is ucus on the new token and the current context. It improves speed only when the stored vectors fit memory and the implementation keeps the buffer layout predictable.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 reduce repeated work and stop honest without repeated stens model files, tokenizer, stop cleanly. loop. Explain why a cache matters for text generation. The cache stores model files are generation.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 when prompt, needed across many tokens. The runtime stores reusable attention vectors, updates them for the newest token, and avoids rebuilding everything from scratch.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 It should compact next-token redanswed for the newest token, and aceces of generation. A useful report says how many tokens were generated, how much memory the cache used, and whether the output still matches reference behavior.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The output head prompt, and local generationixe hardware relarmicte model on a 486ele model on a 486? text generated, how musufoduct ming th memory the cache used, and whether the output still matches reference behavior.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 that le, feed-forward layers, normalization, and logits with predictable memory and repeatable behavior. GPT2 BASIC on a 486 should answer in terms of memory, fixed-point arithmetic, model files, and decode-loop timing.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 Cache reuse saves repeated attention work over while rededestable. Explain why a cache matters for text generation. Cache reuse does not mange the trained weights or make the model smarter.
```

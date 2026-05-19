# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.736`
Prompt pass rate: `2/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.940 | 2 | 0.0% | 2 | 0 | no | RETRAIN |
| 486_target | 0.820 | 1 | 0.0% | 1 | 0 | no | RETRAIN |
| dos_model | 0.760 | 1 | 0.0% | 1 | 0 | yes | PASS |
| basic_runtime | 0.673 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| optimization | 0.739 | 1 | 0.0% | 1 | 0 | yes | PASS |
| heldout_cache | 0.819 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_timing | 0.690 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_limits | 0.591 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_fixed_point | 0.697 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.635 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, and sample b
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 inskery, weights, and the model should talk about file comparing continues that transform
```

### dos_model

Prompt: `DOS language models need`

```text
 to reduce with relevant work from trained weights.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 run on a 486. A ushould asestion and cacaume chontes an measured output.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 profile comparison profile usefuncta/d.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 cache, a correct answer should say that The model should talk about file comparing constr
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 Beconde largement by a computing computed on the prompt as producing the cache, answare,
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The ub useses umpreding report.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 and from lobehavior is concrete and measured around that transformer models are neral key
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The BASIC useful continuation, and language model.
```

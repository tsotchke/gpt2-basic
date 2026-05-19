# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.650`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.690 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_timing | 0.630 | 0 | 16.7% | 1 | no | RETRAIN |
| heldout_limits | 0.665 | 0 | 5.6% | 1 | no | RETRAIN |
| heldout_fixed_point | 0.665 | 0 | 7.7% | 1 | no | RETRAIN |
| heldout_profiles | 0.599 | 0 | 27.3% | 1 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 of the computing that the come also the come a stemplation of the come and the come alsur
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The the come a constrained of the computing the transformer the computing the transformer
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The trained the comes are the model a stery to the come the come a sterious of the come a
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 of the trained the computing the transformer the computing the computed to the consider t
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The computing the transformer the cometinue also the computing the transformer the constr
```


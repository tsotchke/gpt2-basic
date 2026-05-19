# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.670`
Prompt pass rate: `0/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.692 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| 486_target | 0.702 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| dos_model | 0.671 | 0 | 6.7% | 1 | 0 | no | RETRAIN |
| basic_runtime | 0.670 | 0 | 5.9% | 1 | 0 | no | RETRAIN |
| optimization | 0.671 | 0 | 6.7% | 1 | 0 | no | RETRAIN |
| heldout_cache | 0.690 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_timing | 0.644 | 0 | 16.7% | 1 | 0 | no | RETRAIN |
| heldout_limits | 0.669 | 0 | 5.6% | 1 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.671 | 0 | 7.7% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.619 | 0 | 27.3% | 1 | 0 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The come alsure a prompt a computing the the come a stemplications of the computed the co
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 procuments - Compressent Computation Conference Computation Conclusion The primalime shou
```

### dos_model

Prompt: `DOS language models need`

```text
 the transformer the computing the the come a sterith the come also the come a sterind the
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 on the trained the come and the come a sterible and the could the modern the the come and
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 the trained the come and the computing that the come a steriouse and the come a stemplary
```

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

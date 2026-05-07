# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.678`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.692 | 0 | 0.0% | 1 | no | RETRAIN |
| 486_target | 0.702 | 0 | 0.0% | 2 | no | RETRAIN |
| dos_model | 0.665 | 0 | 6.7% | 1 | no | RETRAIN |
| basic_runtime | 0.665 | 0 | 5.9% | 1 | no | RETRAIN |
| optimization | 0.665 | 0 | 6.7% | 1 | no | RETRAIN |

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


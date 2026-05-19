# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.676`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.689 | 0 | 0.0% | 2 | no | RETRAIN |
| 486_target | 0.680 | 0 | 0.0% | 2 | yes | RETRAIN |
| dos_model | 0.698 | 0 | 0.0% | 2 | no | RETRAIN |
| basic_runtime | 0.624 | 0 | 0.0% | 1 | yes | RETRAIN |
| optimization | 0.692 | 0 | 0.0% | 1 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The Contig innovation ofts, ecorrte meain profile, quality, speed, and then tie them to m
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 dems, preducing the ker(6 sult, and sample profile arrly: pemons, compute-2.
```

### dos_model

Prompt: `DOS language models need`

```text
s a shold eenfor loop. The implementation produces earl for attention exponentials, and sa
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 with the result should repofile gones arther.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 system. The tadetes stoker only inference with timing, seconds, tokens, measured, hardwar
```


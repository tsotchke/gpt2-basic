# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.793`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.699 | 0 | 0.0% | 1 | no | RETRAIN |
| 486_target | 0.814 | 1 | 0.0% | 2 | no | PASS |
| dos_model | 0.696 | 0 | 0.0% | 2 | no | RETRAIN |
| basic_runtime | 0.940 | 2 | 0.0% | 2 | no | PASS |
| optimization | 0.817 | 1 | 0.0% | 2 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The generates the target hardware. The is continues weight resould compatibile parationhm
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 is truntime. The model should show then logits, and amy small, hike fixed-point phases st
```

### dos_model

Prompt: `DOS language models need`

```text
s a scaled have is thare in transformer architecture al inference: it is the runtime canno
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses arrays for tokens, embeddings, reuse buffers, and prameter compt only the arent to i
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 timing report. The is concrete, and measured asking screening, sepating seped, memory dis
```


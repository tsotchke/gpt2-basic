# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.769`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.690 | 0 | 0.0% | 1 | yes | RETRAIN |
| 486_target | 0.770 | 1 | 7.7% | 1 | no | PASS |
| dos_model | 0.693 | 0 | 0.0% | 2 | no | RETRAIN |
| basic_runtime | 0.873 | 4 | 0.0% | 2 | yes | PASS |
| optimization | 0.822 | 1 | 0.0% | 2 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The BASIC useful continuation core instead of drifting into generic prose.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
-2-min 486sx-safe -perf 486sx-safe -perf becosion real inference path run: timing report, 
```

### dos_model

Prompt: `DOS language models need`

```text
s a small vocan with the a tradeoff between useful text, fixed, and len the hardware revid
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses fixed aroche dat bete tokens, loops, model.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 evidence. The measurable target is better quality without losing speed, memory discipline
```


# GPT2-BASIC Quality Report

Model profile: `486sx-safe-subword-prototype`
Shape: `2L 48D 4H ctx192 hidden192 vocab512`
Evaluation backend: `subword-host`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.713`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.488 | 0 | 0.0% | 3 | no | RETRAIN |
| 486_target | 0.852 | 2 | 17.6% | 2 | yes | PASS |
| dos_model | 0.647 | 0 | 0.0% | 2 | no | RETRAIN |
| basic_runtime | 0.669 | 0 | 5.3% | 2 | yes | RETRAIN |
| optimization | 0.911 | 2 | 0.0% | 2 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 -for, (2. --- 333), 292016), -2), -- 2), 486-3., 
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 2--6 486 486 486  can  can  can  can 486 weights  can  can 6 weights 486 weights  can  can .
```

### dos_model

Prompt: `DOS language models need`

```text
 vectors, and  correct answer should say that 1. The 486 486  can 486  can  can 2 , 386 486 386 1
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
: GPT2.EXE --per this GPT2.EXE -per.2 path. for 486-per. -vectors 4 uses  physical inference seconds, 486 486 1.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
, profile, quality, speed, memory, measure, tradeoff. The model 486 486  byte prompt 6 386 486: GPT2.EXE -per10 -per10
```


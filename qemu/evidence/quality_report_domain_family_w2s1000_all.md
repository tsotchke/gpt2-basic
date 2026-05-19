# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.775`
Prompt pass rate: `3/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.600 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| 486_target | 0.814 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| dos_model | 0.697 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.933 | 2 | 0.0% | 2 | 1 | no | RETRAIN |
| optimization | 0.810 | 1 | 0.0% | 2 | 1 | no | RETRAIN |
| heldout_cache | 0.843 | 1 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.597 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_limits | 0.668 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_fixed_point | 0.849 | 2 | 0.0% | 1 | 0 | yes | PASS |
| heldout_profiles | 0.943 | 4 | 0.0% | 3 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The generates the target hardware.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 is truntime. The model should show then logits, and amy small, hike fixed-point phases st
```

### dos_model

Prompt: `DOS language models need`

```text
 explain obehimat sill, hardware perf transformers are model estrained vectors the next by
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

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 cache fortery computing constraints force a tradeoff between model predicts compatin.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 This uservid belock menthat port.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The The target behavior is concrete, and tied to namy trioun on an 486.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 on of loweight fixed-point runtime.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The BASIC, arrrays, fixed, and then tie them to speed, memory, measure, tradeoff.
```

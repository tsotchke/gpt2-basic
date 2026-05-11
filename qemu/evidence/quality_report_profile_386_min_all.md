# GPT2-BASIC Quality Report

Model profile: `386-min`
Shape: `2L 32D 4H ctx128 hidden128 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.614`
Prompt pass rate: `0/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.610 | 0 | 37.5% | 2 | 0 | no | RETRAIN |
| 486_target | 0.606 | 0 | 43.8% | 1 | 0 | no | RETRAIN |
| dos_model | 0.614 | 0 | 57.1% | 1 | 0 | no | RETRAIN |
| basic_runtime | 0.604 | 0 | 55.0% | 1 | 0 | no | RETRAIN |
| optimization | 0.607 | 0 | 50.0% | 1 | 0 | no | RETRAIN |
| heldout_cache | 0.607 | 0 | 33.3% | 1 | 0 | no | RETRAIN |
| heldout_timing | 0.670 | 0 | 6.2% | 1 | 0 | no | RETRAIN |
| heldout_limits | 0.607 | 0 | 33.3% | 1 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.607 | 0 | 33.3% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.610 | 0 | 31.2% | 1 | 0 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The implementation atter the the the the the the trat the the the the the te contraine th
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 trained the the te the the the the the the transformer the the the the the the contraine
```

### dos_model

Prompt: `DOS language models need`

```text
 the the the the the transformer the the the the transformer the the transformer ainter al
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 and the the the the the the the the the the trardware the the the the the the te the te c
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 the the the the the the the the trardware the the the the the the te the te contrainal th
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 a the the the the the the transformer the the the the the te the te contrainal al and tha
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The is implementation arined in the the in the the the contran the te computing and the a
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The is the the the the the the transformer the the the the the te the te contrained al an
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 or the the the the the the transformer the the the the the te the te contrained al and th
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The implementation arined the the the the the the the the te the te contrained al and tha
```

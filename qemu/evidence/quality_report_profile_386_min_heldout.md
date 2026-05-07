# GPT2-BASIC Quality Report

Model profile: `386-min`
Shape: `2L 32D 4H ctx128 hidden128 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.605`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.587 | 0 | 33.3% | 1 | no | RETRAIN |
| heldout_timing | 0.665 | 0 | 6.2% | 1 | no | RETRAIN |
| heldout_limits | 0.587 | 0 | 33.3% | 1 | no | RETRAIN |
| heldout_fixed_point | 0.594 | 0 | 37.5% | 1 | no | RETRAIN |
| heldout_profiles | 0.590 | 0 | 31.2% | 1 | no | RETRAIN |

## Generated Continuations

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
d or the the the the the the the transformer ansformer the ardware the the the the contril
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The implementation arined the the the the the the the the te the te contrained al and tha
```


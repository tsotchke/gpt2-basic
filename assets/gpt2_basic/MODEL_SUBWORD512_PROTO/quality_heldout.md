# GPT2-BASIC Quality Report

Model profile: `486sx-safe-subword-prototype`
Shape: `2L 48D 4H ctx192 hidden192 vocab512`
Evaluation backend: `subword-host`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.166`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.166 | 0 | 0.0% | 0 | 1 | no | RETRAIN |
| heldout_timing | 0.166 | 0 | 0.0% | 0 | 1 | no | RETRAIN |
| heldout_limits | 0.166 | 0 | 0.0% | 0 | 1 | no | RETRAIN |
| heldout_fixed_point | 0.166 | 0 | 0.0% | 0 | 1 | no | RETRAIN |
| heldout_profiles | 0.166 | 0 | 0.0% | 0 | 1 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text

```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text

```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text

```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text

```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text

```

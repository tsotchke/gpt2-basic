# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.655`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.694 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_timing | 0.541 | 0 | 12.5% | 2 | yes | RETRAIN |
| heldout_limits | 0.706 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_fixed_point | 0.701 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_profiles | 0.632 | 0 | 0.0% | 1 | yes | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 loop. -lin QEMU complementation Approach : Sparse representation B it cabufe insteal data
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 GPT2.EXE C: GPT2.EXE C: MODEL --perf C: PERF.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The ub und on exerstranting da scripationsmud remory understandable algorithms 2.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
d devandipled precision fotenlined-fatence dedsses dis of dsur meconsing relo: DOS program
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 Cexploring This the contract, and to snodames not 486.
```


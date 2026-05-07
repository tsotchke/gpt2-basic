# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab384`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.812`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.696 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_timing | 0.670 | 0 | 0.0% | 8 | no | RETRAIN |
| heldout_limits | 0.805 | 1 | 0.0% | 2 | yes | PASS |
| heldout_fixed_point | 0.948 | 2 | 0.0% | 1 | no | PASS |
| heldout_profiles | 0.941 | 3 | 0.0% | 3 | no | PASS |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
arger should bgoout the model is cont the DOS BASIC, thexplain adarined on the DOS program emory, and sembedd throadw-ble normedds
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 Woullllllllmwe 2. -pnicatual BASIC, we mememory trained decode cache vectors reduce inference weights, 
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
e 4 486 486detrse chniques malimplements within loop hardware limitations.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 arithmetic cost, and fixed point math the model production eded to tokens, explainic storight abic nvalualid memor
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 the that thestiming through: 1. Transunivalll for allogits, whosen explainic, quality, speed, memory, me
```


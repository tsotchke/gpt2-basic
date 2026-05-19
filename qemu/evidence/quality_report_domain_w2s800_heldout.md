# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.708`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.819 | 1 | 0.0% | 2 | no | PASS |
| heldout_timing | 0.696 | 0 | 0.0% | 3 | no | RETRAIN |
| heldout_limits | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_fixed_point | 0.697 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_profiles | 0.635 | 0 | 0.0% | 1 | yes | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 cache, a correct answer should say that The model should talk about file comparing constr
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
If ust not a cloop behins, and reducation conn ditrocusofion mainsting sto smalll byte s c
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The ub useses umpreding report. The measurable target is better quality without losing sp
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 and from lobehavior is concrete and measured around that transformer models are neral key
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The BASIC useful continuation, and language model.
```


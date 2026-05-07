# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab384`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.726`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.939 | 6 | 0.0% | 2 | no | PASS |
| heldout_timing | 0.694 | 0 | 0.0% | 3 | no | RETRAIN |
| heldout_limits | 0.665 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_fixed_point | 0.689 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_profiles | 0.643 | 0 | 0.0% | 1 | yes | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 cache, context, tokens, speed, memory, reuse instead of drifting into generic prose. When ren ren reduchis memory, context, speed, weights, small, 
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 cont, the key point is this: A logragrits, answorrre needurance scaled-decomputation easkerns. A whork is not the
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The un this2. Optimized consting for the maconved alguage models?
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
d comentration alsers, 27. Reucon des betraning ipror: Asronte, nth the trained work is not the n answer is that prused no
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 Compronest nempratly tend computation computation.
```


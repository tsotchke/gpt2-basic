# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.667`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.711 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_timing | 0.648 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_limits | 0.660 | 0 | 0.0% | 2 | yes | RETRAIN |
| heldout_fixed_point | 0.623 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_profiles | 0.694 | 0 | 0.0% | 2 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 vidence depends on this: The real model profile comparison an 486s deminst argethe det.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
meK cont text operas read is the DOS fixed-point runtime.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 T2 BASIC, arrays, fixed, lear tradeoff between model mitrations.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 : The smodel is the project det a printable byte.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The uses under should connect finclue with the relevant words are a bud trixed withoins t
```


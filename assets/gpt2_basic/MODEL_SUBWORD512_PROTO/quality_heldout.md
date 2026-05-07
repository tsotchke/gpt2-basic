# GPT2-BASIC Quality Report

Model profile: `486sx-safe-subword-prototype`
Shape: `2L 48D 4H ctx192 hidden192 vocab512`
Evaluation backend: `subword-host`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.673`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.938 | 3 | 0.0% | 2 | no | PASS |
| heldout_timing | 0.649 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_limits | 0.509 | 0 | 20.0% | 1 | no | RETRAIN |
| heldout_fixed_point | 0.587 | 0 | 0.0% | 9 | no | RETRAIN |
| heldout_profiles | 0.683 | 0 | 0.0% | 2 | yes | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
, and tokens per second. The answer should  cannot per speed, uses key and seconds. The answer hardware per this runtime  short context  speed, 1 into 1 into arithmetic inference 33
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 8 include host for when 6 6 byte 6 6, 532. 486, the practical rule  constraints, 53. 486, 486, 
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 into . - - - 3 3- 3 3- 3 3 4 about fixed-point runtime: 2. 386 486 486 486
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 hardware, and 2. - comes from . 3. -2 --vectors  inside the 1(2.5555555553.555550
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 . The answer should then  per fixed-point correctness. 486, the 1(86 486 1( about -1336).
```


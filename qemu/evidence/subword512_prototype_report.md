# GPT2-BASIC Subword Host Prototype

This is a host-only architecture probe. The transformer is real, but the checkpoint is not DOS-ready until the subword tokenizer is implemented in the production runtime.

## Training

- Output: `assets/gpt2_basic/MODEL_SUBWORD512_PROTO`
- Profile: `486sx-safe`
- Vocab size: 512
- Subword pieces: 254
- Training tokens: 0
- Core docs: 0
- External docs: 0

## Held-out Quality

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


## Runtime-Regression Quality

# GPT2-BASIC Quality Report

Model profile: `486sx-safe-subword-prototype`
Shape: `2L 48D 4H ctx192 hidden192 vocab512`
Evaluation backend: `subword-host`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.713`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.488 | 0 | 0.0% | 3 | no | RETRAIN |
| 486_target | 0.852 | 2 | 17.6% | 2 | yes | PASS |
| dos_model | 0.647 | 0 | 0.0% | 2 | no | RETRAIN |
| basic_runtime | 0.669 | 0 | 5.3% | 2 | yes | RETRAIN |
| optimization | 0.911 | 2 | 0.0% | 2 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 -for, (2. --- 333), 292016), -2), -- 2), 486-3., 
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 2--6 486 486 486  can  can  can  can 486 weights  can  can 6 weights 486 weights  can  can .
```

### dos_model

Prompt: `DOS language models need`

```text
 vectors, and  correct answer should say that 1. The 486 486  can 486  can  can 2 , 386 486 386 1
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
: GPT2.EXE --per this GPT2.EXE -per.2 path. for 486-per. -vectors 4 uses  physical inference seconds, 486 486 1.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
, profile, quality, speed, memory, measure, tradeoff. The model 486 486  byte prompt 6 386 486: GPT2.EXE -per10 -per10
```

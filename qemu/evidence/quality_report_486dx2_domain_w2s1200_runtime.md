# GPT2-BASIC Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab258`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.741`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.690 | 0 | 0.0% | 1 | no | RETRAIN |
| 486_target | 0.802 | 1 | 0.0% | 2 | yes | PASS |
| dos_model | 0.699 | 0 | 0.0% | 1 | no | RETRAIN |
| basic_runtime | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |
| optimization | 0.818 | 1 | 0.0% | 2 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The fixed-point inference. The measurable inference, the key point is this: The cache sav
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
. Subatimes a stim all about ptilememory whith implementation count.
```

### dos_model

Prompt: `DOS language models need`

```text
s vemory contraioner. QEMU log alits profile compares can run on algithm araithmed repence
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 whille target is now claim temight pincacal osese fundamental vive olid the contract, and
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 profile, the practical rule is that The measurace ntinuable ttives ureor funded menstive 
```


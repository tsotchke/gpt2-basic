# GPT2-BASIC Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.721`
Prompt pass rate: `0/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.690 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| 486_target | 0.824 | 1 | 0.0% | 1 | 0 | no | RETRAIN |
| dos_model | 0.692 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| basic_runtime | 0.695 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| optimization | 0.818 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_cache | 0.708 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_timing | 0.697 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_limits | 0.695 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.694 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.695 | 0 | 0.0% | 1 | 0 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The fixed-point inference. The measurable inference, the key point is this: The cache sav
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 sand computing weights predictable integes and runs arithmetic with the rimitather, coste
```

### dos_model

Prompt: `DOS language models need`

```text
 vemory context (as has real hardware tansformer betwork tokens, comeses, and tied to ande
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

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 begins. The answer should be short, concrete, and tied to the DOS fixed-point runtime.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 C: The implementation surpeses and model princeXEMUndating and this: model strained weigh
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 This program ays abut constrained on embeddings, causal self attention lo enoughid in the
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 stornes. A real sucor pabot aloce and PC constraints trainted model says FPU model conver
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 C DOS transformer opt, model insteger arithmetic with the relevant words metatching his c
```

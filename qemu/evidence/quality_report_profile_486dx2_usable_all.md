# GPT2-BASIC Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.685`
Prompt pass rate: `0/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.690 | 0 | 0.0% | 3 | 0 | no | RETRAIN |
| 486_target | 0.643 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| dos_model | 0.822 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.618 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| optimization | 0.823 | 1 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_cache | 0.591 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_timing | 0.640 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_limits | 0.695 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.694 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.631 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The generated a ta tthe checent aceled a ben on emin dinstead the innned the fundamental
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 not with cleystey to wown that the cactual be ge models?
```

### dos_model

Prompt: `DOS language models need`

```text
 now nerervintag hardware haperf hidw loops DOS, the model training fixed-point arithmetic
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 with concentrational wave drevistation.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 profile dathed reploying thange modelsthed implementation and preceBASIC The should stey
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 wat whith the corece acturable.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The quesestions tures that the core from contraints.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The gas as trained a transformers in an rinit emsight is that transformer models computin
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 instead din decades. It nshe the the contraby refrom the model computes context window th
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The quantantization and forwardinationation.
```

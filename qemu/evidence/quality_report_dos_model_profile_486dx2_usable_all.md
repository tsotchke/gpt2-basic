# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab258`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `all`
Source log: `<repo>/qemu/evidence/quality_486_model_profile_486dx2_usable.log`
Quality status: `NEEDS_TRAINING`
Average score: `0.715`
Prompt pass rate: `2/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.692 | 0 | 0.0% | 3 | no | RETRAIN |
| 486_target | 0.623 | 0 | 0.0% | 2 | yes | RETRAIN |
| dos_model | 0.948 | 2 | 0.0% | 5 | no | PASS |
| basic_runtime | 0.669 | 0 | 0.0% | 1 | yes | RETRAIN |
| optimization | 0.852 | 1 | 0.0% | 1 | yes | PASS |
| heldout_cache | 0.704 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_timing | 0.642 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_limits | 0.677 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_fixed_point | 0.696 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_profiles | 0.643 | 0 | 0.0% | 1 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
The generated a ta tthe checent aceled a ben on emin dinstead the innned the fundamental
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
. Bethen ot and model is hardware ccontraints.
```

### dos_model

Prompt: `DOS language models need`

```text
stand weights. The could be plains simplementation sembeddings, runstttimedddddental endar
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
with clear beful on demonstratin that the functinal and a prompt.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
, profiled the the inster finction transformers models in the fundamental porach promater.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
wat whith the corece acturfomer, mel model training that the fund model hardware.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
The quesestions tures that the core from contraints.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
The gas as trained a transformers in an rinitruce implemented to and.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
s instead disings, and the trad2 penticising withth project by- complented argrate bufor d
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
The quantantization and forwardinationation. izer.
```


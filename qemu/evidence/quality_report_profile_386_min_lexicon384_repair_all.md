# GPT2-BASIC Quality Report

Model profile: `386-min`
Shape: `2L 32D 4H ctx128 hidden128 vocab384`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.714`
Prompt pass rate: `2/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.670 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| 486_target | 0.601 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| dos_model | 0.821 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.947 | 2 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.747 | 1 | 0.0% | 1 | 0 | yes | PASS |
| heldout_cache | 0.620 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| heldout_timing | 0.630 | 0 | 0.0% | 3 | 0 | yes | RETRAIN |
| heldout_limits | 0.663 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.800 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_profiles | 0.637 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The demple: pets tokenic stays conly imprec areferl paragrapsess 3.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 prodect.bas GPT2SRC MAIN.BAS LABMAIN.
```

### dos_model

Prompt: `DOS language models need`

```text
 prompt-head short be project, but buffer on real hardware, reun transformer mode emode theck britet byte algoithne thi
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 keeps tokens, model weights, logits, and sampling budge byCas, and but read byte.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 profile explit produce ather answing model.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 stemile port, but buffers, and read-buding the 486.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 Phatinul apppling product-achit shable report.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 If P386-2/6 in BASIC86.macx-ed2 486dx2-usable 486dx2-usable 486dx4-plus pents/gencen contal re
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 algorithms are than maching, 20160-2 weights nout,1303 40048 output version with sy weights, comping bufer pare o
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 They-2 BASIC ontifer projecing onterat: Simic-pacRAS.
```

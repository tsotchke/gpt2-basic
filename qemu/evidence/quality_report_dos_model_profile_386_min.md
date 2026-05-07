# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `386-min`
Shape: `2L 32D 4H ctx128 hidden128 vocab258`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `runtime-regression`
Source log: `/Users/tyr/Desktop/gpt2-basic/qemu/evidence/quality_486_model_profile_386_min.log`
Quality status: `NEEDS_TRAINING`
Average score: `0.645`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.859 | 2 | 21.4% | 1 | no | PASS |
| 486_target | 0.589 | 0 | 43.8% | 1 | no | RETRAIN |
| dos_model | 0.595 | 0 | 57.1% | 1 | no | RETRAIN |
| basic_runtime | 0.592 | 0 | 37.5% | 1 | no | RETRAIN |
| optimization | 0.588 | 0 | 50.0% | 1 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
The implementation ater the the the the transformer the the the the contre trained the th
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
trained the the te the the the the the the transformer the the the the the the contraine
```

### dos_model

Prompt: `DOS language models need`

```text
the the the the the transformer the the the the transformer the the transformer ainter al
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
and the the the the the the transformer the ardware the the the the the contraine ter the
```

### optimization

Prompt: `To improve performance on real hardware`

```text
the the the the the the the the trardware the the the the the the te the te contrainal th
```


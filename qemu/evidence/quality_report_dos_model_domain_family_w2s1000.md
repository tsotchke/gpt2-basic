# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `runtime-regression`
Source log: `/Users/tyr/Desktop/gpt2-basic/qemu/evidence/quality_486_model_domain_family_w2s1000.log`
Quality status: `NEEDS_TRAINING`
Average score: `0.763`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.666 | 0 | 0.0% | 1 | yes | RETRAIN |
| 486_target | 0.815 | 1 | 0.0% | 2 | no | PASS |
| dos_model | 0.821 | 1 | 0.0% | 1 | no | PASS |
| basic_runtime | 0.697 | 0 | 0.0% | 2 | no | RETRAIN |
| optimization | 0.818 | 1 | 0.0% | 2 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
The generates the target hardware. The is continues without DIC.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
is truntime. The model should show then logits, and amy small, hike fixed-point phases st
```

### dos_model

Prompt: `DOS language models need`

```text
s a scaled have is thare in the genereary coun with the memory benverations the sto checkp
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
reta on real hardware timing report. The measurable target is better quality without losi
```

### optimization

Prompt: `To improve performance on real hardware`

```text
timing report. The is concrete, and measured asking screening, sepating seped, memory dis
```


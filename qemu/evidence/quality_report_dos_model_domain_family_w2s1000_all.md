# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `all`
Source log: `<repo>/qemu/evidence/quality_486_model_domain_family_w2s1000.log`
Quality status: `NEEDS_TRAINING`
Average score: `0.806`
Prompt pass rate: `7/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.666 | 0 | 0.0% | 1 | yes | RETRAIN |
| 486_target | 0.815 | 1 | 0.0% | 2 | no | PASS |
| dos_model | 0.821 | 1 | 0.0% | 1 | no | PASS |
| basic_runtime | 0.697 | 0 | 0.0% | 2 | no | RETRAIN |
| optimization | 0.818 | 1 | 0.0% | 2 | no | PASS |
| heldout_cache | 0.844 | 1 | 0.0% | 2 | yes | PASS |
| heldout_timing | 0.899 | 2 | 0.0% | 1 | yes | PASS |
| heldout_limits | 0.639 | 0 | 0.0% | 2 | yes | RETRAIN |
| heldout_fixed_point | 0.920 | 2 | 0.0% | 1 | yes | PASS |
| heldout_profiles | 0.945 | 4 | 0.0% | 3 | yes | PASS |

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

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
cache fortery computing constraints force a tradeoff between model predicts compatin.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
IC C: C: GPT2.EXE C: MODEL qemu/evidence/hardware perf report.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
The The tare is them firstic and be it is correctness.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
on of loweight fixed-point runtime. The model must atch the result.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
The BASIC, arrrays, fixed, and then tie them to speed, memory, measure, tradeoff.
```


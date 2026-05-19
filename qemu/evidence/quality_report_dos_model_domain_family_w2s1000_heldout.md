# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `heldout`
Source log: `<repo>/qemu/evidence/quality_486_model_domain_family_w2s1000.log`
Quality status: `NEEDS_TRAINING`
Average score: `0.850`
Prompt pass rate: `4/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.844 | 1 | 0.0% | 2 | yes | PASS |
| heldout_timing | 0.899 | 2 | 0.0% | 1 | yes | PASS |
| heldout_limits | 0.639 | 0 | 0.0% | 2 | yes | RETRAIN |
| heldout_fixed_point | 0.920 | 2 | 0.0% | 1 | yes | PASS |
| heldout_profiles | 0.945 | 4 | 0.0% | 3 | yes | PASS |

## Generated Continuations

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


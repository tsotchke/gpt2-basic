# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab258`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `heldout`
Source log: `/Users/tyr/Desktop/gpt2-basic/qemu/evidence/quality_486_model_profile_486dx2_usable.log`
Quality status: `NEEDS_TRAINING`
Average score: `0.673`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.704 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_timing | 0.642 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_limits | 0.677 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_fixed_point | 0.696 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_profiles | 0.643 | 0 | 0.0% | 1 | yes | RETRAIN |

## Generated Continuations

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


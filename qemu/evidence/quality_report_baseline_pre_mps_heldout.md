# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.681`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.686 | 0 | 0.0% | 2 | yes | RETRAIN |
| heldout_timing | 0.691 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_limits | 0.685 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_fixed_point | 0.693 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_profiles | 0.651 | 0 | 0.0% | 1 | yes | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
, feed-forward computation, logits, and sampling from trained parameters.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The be Itre mise asmisscdation an clained on the his accecept trriorits ted o completion 
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The user should boot DOS, run GPT2.EXE, enter a prompt, and see the trained model continu
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
s instead of reading a canned response. It names the prompt, explains the machine, and con
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 It camats bout be real hard from an ordinary FAT directory.
```


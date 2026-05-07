# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab384`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.748`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.716 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_timing | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_limits | 0.696 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_fixed_point | 0.693 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_profiles | 0.942 | 2 | 0.0% | 2 | no | PASS |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
, and nofrom am modern A.36 C: MODEL CAURS MAIFEST.json qemu/evidence/online training data audit.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 or plain DOS-orionlisimings and : The getarget is concleneration provire ofue without fficremodern ffern ferited itas 2. In the st s transformer can run ininfas
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 Thes t A tin iretroot and implemical pucal performance guation al- , ardamintoken could havented des more elier 2. Optimization techniques devew
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
. A useful continuation connects cache, context, tokens, speed, memory, reuse instead of drifting into generic prose. In the architecture sweep, a correct answer should 
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The generation eering rom ipemation timing, seconds, tokens, and then tie them to measured, hardware, QEMU. A QEMU result should say which CPU profile, ic
```


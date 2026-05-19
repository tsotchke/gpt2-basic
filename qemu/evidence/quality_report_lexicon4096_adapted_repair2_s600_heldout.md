# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.743`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.694 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_timing | 0.696 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_limits | 0.937 | 6 | 0.0% | 2 | 1 | no | RETRAIN |
| heldout_fixed_point | 0.694 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.693 | 0 | 0.0% | 1 | 0 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 prow prort pror prort siming prer pror pror sere pran proratimpr pror pror pror seror ser
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 486d integer peded steral peropr deratin dere cont sincation mplent pror deroprt seration sede peco
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 stopwatch numbers are not hardware claims; the authoritative record is the PERF log emitted by the DOS executable. The answer should be short, concrete, and tied to the DOS fixed-point runtime. In the production inference path, production evidence depends on this: The model must stay small enough to load its weights and work buffers before generation begins. A short response about decode cache should include memory, context, speed, and then tie them to weights, small, ha
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 prow pror propt seror pror sing pror sicat bange Cors peratin pron sere mpratin pror sere
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 ment eral pror prorect dere decos preratin sede pedecor deder coumpres prerat pror prort 
```


# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.768`
Prompt pass rate: `1/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.647 | 0 | 14.8% | 2 | 0 | no | RETRAIN |
| 486_target | 0.695 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| dos_model | 0.943 | 2 | 0.0% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.973 | 6 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.698 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_cache | 0.694 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_timing | 0.696 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_limits | 0.944 | 6 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.694 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.694 | 0 | 0.0% | 1 | 0 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 486, raited phases still match the exported host checkpoint closely enough to trust the DOS result. Inside the 486 decode loop, the key point is this: A useful timing row separates model estimates from measured emulator or physical hardware evidence. The target behavior is concrete and measured around timing, seconds, tokens, measured, hardware, the emulator. A useful timing row separates model estimates from measured emulator or physical hardware evidence. A2 generation cache reuses key and value vectors for tokens that are already inside
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 semint ence/ence/ress peration pede propr deratin sere pran proration sere seran teder mu
```

### dos_model

Prompt: `DOS language models need`

```text
 short context), 2 BASIC transformer embeddings, cache vectors, logits, and weights, contion, an, and integer inference path, sweardware, the emulator. The DOS executable should report generated tokens, elapsed seconds, and tokens per second from inside the timed decode loop. The report must include enough context to repeat the run: mode
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 loads trained weights and computes the next byte through the transformer runtime should stop instead of pretending if the trained model files are absent. The model directory is the contract: the model configuration describes shape, the fixed weight file stores fixed weights, and the exponential lookup table stores the exp table. The target behavior is concrete and measured around BASIC, arrays, fixed, tokens, loops, model.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 reduce repeated divisions, keareofins sepe sepe pratic/ration sent peron sint peron sint sent peror secor secon si
```

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
 sere mint ence/esin pede pror pratin pror deration sere proprt deration sing sere sere mp
```

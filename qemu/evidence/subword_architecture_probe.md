# GPT2-BASIC Subword Architecture Probe

This probe estimates whether a compact domain subword vocabulary is worth implementing before more fine-tuning. It does not change the DOS runtime yet.

Corpus basis: `data/domain_curriculum/domain_curriculum.txt`

## Result

| Vocab | Sample tokens | Bytes | Token/byte | Params | Runtime MB | Est 486SX tok/s | Est 486DX2 tok/s | Est 70-byte seconds on 486DX2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 258 | 1100 | 1100 | 1.000 | 90882 | 0.497 | 1.36 | 5.42 | 12.9 |
| 384 | 531 | 1100 | 0.483 | 103104 | 0.545 | 1.45 | 5.80 | 5.9 |
| 512 | 384 | 1100 | 0.349 | 115520 | 0.594 | 1.42 | 5.66 | 4.2 |
| 768 | 312 | 1100 | 0.284 | 140352 | 0.691 | 1.32 | 5.27 | 3.8 |
| 1024 | 286 | 1100 | 0.260 | 165184 | 0.789 | 1.22 | 4.88 | 3.7 |

## Interpretation

The best planning point in this probe is vocab `1024`. It reduces the sample token count to 26.0% of byte-level while keeping runtime memory at 0.789 MB.

The quality reason is stronger than the speed reason: a domain subword vocabulary lets the model emit whole pieces such as ` fixed-point`, ` runtime`, ` tokens`, and ` memory` instead of relearning spelling byte by byte.

## Example Pieces

- `258`: 
- `384`: ` the`, ` and`, `the`, ` The`, `and`, ` should`, `The`, ` tokens`, ` fixed-point`, ` memory`, ` speed`, ` weights`, `should`, ` answer`, `tokens`, ` model`, ` profile`, `fixed-point`, ` runtime`, ` DOS`, `memory`, `speed`, ` context`, `weights`, ` production`, ` cache`, ` logits`, ` quality`, ` that`, ` hardware`
- `512`: ` the`, ` and`, `the`, ` The`, `and`, ` should`, `The`, ` tokens`, ` fixed-point`, ` memory`, ` speed`, ` weights`, `should`, ` answer`, `tokens`, ` model`, ` profile`, `fixed-point`, ` runtime`, ` DOS`, `memory`, `speed`, ` context`, `weights`, ` production`, ` cache`, ` logits`, ` quality`, ` that`, ` hardware`
- `768`: ` the`, ` and`, `the`, ` The`, `and`, ` should`, `The`, ` tokens`, ` fixed-point`, ` memory`, ` speed`, ` weights`, `should`, ` answer`, `tokens`, ` model`, ` profile`, `fixed-point`, ` runtime`, ` DOS`, `memory`, `speed`, ` context`, `weights`, ` production`, ` cache`, ` logits`, ` quality`, ` that`, ` hardware`
- `1024`: ` the`, ` and`, `the`, ` The`, `and`, ` should`, `The`, ` tokens`, ` fixed-point`, ` memory`, ` speed`, ` weights`, `should`, ` answer`, `tokens`, ` model`, ` profile`, `fixed-point`, ` runtime`, ` DOS`, `memory`, `speed`, ` context`, `weights`, ` production`, ` cache`, ` logits`, ` quality`, ` that`, ` hardware`

## Implementation Consequence

The production tokenizer contract is now wired through host training/export, vector generation, quality evaluation, DOS tokenizer loading, sampler masking, and QEMU staging of `VOCAB.BIN`. This probe remains useful for sizing decisions, but quality promotion still requires a real BPE training sweep with DOS held-out quality, runtime-regression quality, vector parity, and QEMU `--perf` evidence.

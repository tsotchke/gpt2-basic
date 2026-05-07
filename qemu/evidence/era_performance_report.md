# GPT2-BASIC Era Performance Model

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Parameters: `90882`
Estimated runtime memory: `520740` bytes (`0.497` MB)
Validation-only phase trace overhead: `2880` bytes
Demo path: `31` prompt tokens, `70` generated tokens, `100` cached forward calls

## Kernel Work

| Metric | Count |
|---|---:|
| Fixed multiply-accumulate style ops | 7,704,560 |
| Fixed multiply helper calls inlined in attention | 1,010,000 |
| Clamp helper calls inlined in attention | 1,050,400 |
| Fixed multiply helper calls inlined in GELU | 230,400 |
| Clamp helper calls inlined in GELU | 230,400 |
| Exp table lookups | 40,400 |
| Integer sqrt ops | 470 |
| Integer divisions before attention-probability hoist | 523,670 |
| Integer divisions after attention-probability hoist | 79,270 |
| Divisions removed from 70-token demo | 444,400 |
| Weighted work reduction | 46.3% |

## Estimated Real-PC Throughput

| Target | Clock | 70-token demo | Tokens/sec | 100-token equivalent | Notes |
|---|---:|---:|---:|---:|---|
| 386DX/33-class, no FPU | 33 MHz | 103.3 s | 0.68 | 147.5 s | Conservative protected-mode 386 estimate |
| 486SX/25, no FPU | 25 MHz | 51.6 s | 1.36 | 73.8 s | No-FPU 486 integer path |
| 486DX/33 | 33 MHz | 25.8 s | 2.71 | 36.9 s | 486 integer path |
| 486DX2/66 | 66 MHz | 12.9 s | 5.42 | 18.4 s | Common real-PC target |
| 486DX4/100 | 100 MHz | 8.6 s | 8.14 | 12.3 s | Fast 486-class target |
| Pentium 60 | 60 MHz | 8.9 s | 7.91 | 12.6 s | Early Pentium target |
| Pentium 133 | 133 MHz | 4.3 s | 16.27 | 6.1 s | High-end DOS-era Pentium |
| QEMU 486dx2-66 --perf | icount | 11.64 s | 6.01 | 16.6 s | qemu-emulation, 486DX2/66-era throttle, icount shift=4 |
| Host-speed QEMU -cpu 486 --perf | host | 0.55 s | 127.27 | 0.8 s | qemu-emulation, host-speed emulator baseline |

Assumptions: throughput is expressed in weighted fixed-point BASIC work units/sec, with 64-bit integer division priced much higher than fixed multiply/add work. QEMU `--perf` rows are emulator evidence from the DOS executable; physical-board timing still wins over both QEMU and the planning model.

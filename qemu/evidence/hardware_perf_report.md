# GPT2-BASIC Hardware/Emulator Performance Evidence

These rows come from `GPT2.EXE --perf` inside DOS. Under QEMU, they are emulated CPU-profile evidence, not physical-board claims.

## Summary

| Evidence | CPU/Profile | Model | Runs | Tokens | Seconds | Tokens/sec | Runtime bytes | Basis |
|---|---|---|---:|---:|---:|---:|---:|---|
| `perf_486_486dx2-66_model_tokheadq4_prod_probe.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 117 | 55.15 | 2.12 | 974724 | qemu-emulation |
| `perf_486_486dx2-66.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 108 | 43.94 | 2.46 | 2055940 | qemu-emulation |
| `perf_486_486dx2-66_model_headq4_prod_probe.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 125 | 58.88 | 2.12 | 1646404 | qemu-emulation |
| `perf_486_486dx2-66_model_profile_386_min.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `386-min` | 3 | 270 | 21.97 | 12.29 | 259108 | qemu-emulation |
| `perf_486_486dx2-66_model_profile_486dx2_usable.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486dx2-usable` | 3 | 202 | 75.96 | 2.66 | 1088292 | qemu-emulation |
| `perf_486_host.log` | `486` / `host (host-speed emulator baseline)` | `486sx-safe` | 3 | 250 | 1.87 | 133.69 | 520740 | qemu-emulation |

## Run Details

| Evidence | Prompt | Prompt tokens | Generated tokens | Seconds | Tokens/sec | Last token |
|---|---|---:|---:|---:|---:|---:|
| `perf_486_486dx2-66_model_tokheadq4_prod_probe.log` | `real_inference` | 3 | 35 | 16.37 | 2.14 | 48 |
| `perf_486_486dx2-66_model_tokheadq4_prod_probe.log` | `486_target` | 2 | 33 | 15.38 | 2.15 | 65 |
| `perf_486_486dx2-66_model_tokheadq4_prod_probe.log` | `basic_runtime` | 4 | 49 | 23.40 | 2.09 | 313 |
| `perf_486_486dx2-66.log` | `real_inference` | 3 | 35 | 14.11 | 2.48 | 48 |
| `perf_486_486dx2-66.log` | `486_target` | 2 | 40 | 16.04 | 2.49 | 510 |
| `perf_486_486dx2-66.log` | `basic_runtime` | 4 | 33 | 13.79 | 2.39 | 313 |
| `perf_486_486dx2-66_model_headq4_prod_probe.log` | `real_inference` | 3 | 35 | 16.42 | 2.13 | 48 |
| `perf_486_486dx2-66_model_headq4_prod_probe.log` | `486_target` | 2 | 57 | 26.58 | 2.14 | 1245 |
| `perf_486_486dx2-66_model_headq4_prod_probe.log` | `basic_runtime` | 4 | 33 | 15.88 | 2.08 | 313 |
| `perf_486_486dx2-66_model_profile_386_min.log` | `real_inference` | 31 | 90 | 7.69 | 11.70 | 106 |
| `perf_486_486dx2-66_model_profile_386_min.log` | `486_target` | 19 | 90 | 6.87 | 13.10 | 34 |
| `perf_486_486dx2-66_model_profile_386_min.log` | `basic_runtime` | 27 | 90 | 7.41 | 12.15 | 103 |
| `perf_486_486dx2-66_model_profile_486dx2_usable.log` | `real_inference` | 31 | 90 | 34.00 | 2.65 | 34 |
| `perf_486_486dx2-66_model_profile_486dx2_usable.log` | `486_target` | 19 | 46 | 16.86 | 2.73 | 48 |
| `perf_486_486dx2-66_model_profile_486dx2_usable.log` | `basic_runtime` | 27 | 66 | 25.10 | 2.63 | 48 |
| `perf_486_host.log` | `real_inference` | 31 | 70 | 0.55 | 127.27 | 48 |
| `perf_486_host.log` | `486_target` | 19 | 90 | 0.66 | 136.36 | 46 |
| `perf_486_host.log` | `basic_runtime` | 27 | 90 | 0.66 | 136.36 | 110 |

## Runtime Contract

| Field | Value |
|---|---|
| Timed region | `decode_loop_only` |
| Sampling | `greedy_temperature_0` |
| Console progress | `disabled` |
| KV cache | `enabled` |
| Arithmetic | `q20.12_fixed` |

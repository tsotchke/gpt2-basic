# GPT2-BASIC Hardware/Emulator Performance Evidence

These rows come from `GPT2.EXE --perf` inside DOS. Under QEMU, they are emulated CPU-profile evidence, not physical-board claims.

## Summary

| Evidence | CPU/Profile | Model | Runs | Tokens | Seconds | Tokens/sec | Runtime bytes | Basis |
|---|---|---|---:|---:|---:|---:|---:|---|
| `perf_486_host.log` | `486` / `host (host-speed emulator baseline)` | `486sx-safe` | 3 | 108 | 2.48 | 43.55 | 2055940 | qemu-emulation |
| `perf_486_386dx-33.log` | `486,-fpu` / `386dx-33 (386DX/33-class conservative throttle)` | `486sx-safe` | 3 | 108 | 351.90 | 0.31 | 2055940 | qemu-emulation |
| `perf_486_486dx-33.log` | `486` / `486dx-33 (486DX/33-era throttle)` | `486sx-safe` | 3 | 108 | 87.93 | 1.23 | 2055940 | qemu-emulation |
| `perf_486_486dx2-66.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 108 | 44.00 | 2.45 | 2055940 | qemu-emulation |
| `perf_486_486dx2-66_kernel.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 108 | 44.77 | 2.41 | 2055940 | qemu-emulation |
| `perf_486_486dx2-66_model_headq4_prod_probe.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 125 | 58.88 | 2.12 | 1646404 | qemu-emulation |
| `perf_486_486dx2-66_model_profile_386_min.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `386-min` | 3 | 270 | 21.97 | 12.29 | 259108 | qemu-emulation |
| `perf_486_486dx2-66_model_profile_486dx2_usable.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486dx2-usable` | 3 | 202 | 75.96 | 2.66 | 1088292 | qemu-emulation |
| `perf_486_486dx2-66_model_tokheadq4_prod_probe.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 117 | 55.15 | 2.12 | 974724 | qemu-emulation |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 117 | 144.18 | 0.81 | 616324 | qemu-emulation |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `486` / `486dx2-66 (486DX2/66-era throttle)` | `486sx-safe` | 3 | 117 | 145.18 | 0.81 | 616324 | qemu-emulation |
| `perf_486_486dx4-100.log` | `486` / `486dx4-100 (486DX4/100 upper-bound throttle)` | `486sx-safe` | 3 | 108 | 21.98 | 4.91 | 2055940 | qemu-emulation |
| `perf_486_486sx-25.log` | `486,-fpu` / `486sx-25 (486SX/25 no-FPU throttle)` | `486sx-safe` | 3 | 108 | 175.88 | 0.61 | 2055940 | qemu-emulation |
| `perf_486_pentium-133.log` | `pentium` / `pentium-133 (Pentium 133-era throttle)` | `486sx-safe` | 3 | 108 | 10.97 | 9.85 | 2055940 | qemu-emulation |
| `perf_486_pentium-60.log` | `pentium` / `pentium-60 (Pentium 60-era throttle)` | `486sx-safe` | 3 | 108 | 21.97 | 4.92 | 2055940 | qemu-emulation |

## Run Details

| Evidence | Prompt | Prompt tokens | Generated tokens | Seconds | Tokens/sec | Last token |
|---|---|---:|---:|---:|---:|---:|
| `perf_486_host.log` | `real_inference` | 3 | 35 | 0.83 | 42.17 | 48 |
| `perf_486_host.log` | `486_target` | 2 | 40 | 0.88 | 45.45 | 510 |
| `perf_486_host.log` | `basic_runtime` | 4 | 33 | 0.77 | 42.86 | 313 |
| `perf_486_386dx-33.log` | `real_inference` | 3 | 35 | 112.98 | 0.31 | 48 |
| `perf_486_386dx-33.log` | `486_target` | 2 | 40 | 128.30 | 0.31 | 510 |
| `perf_486_386dx-33.log` | `basic_runtime` | 4 | 33 | 110.62 | 0.30 | 313 |
| `perf_486_486dx-33.log` | `real_inference` | 3 | 35 | 28.23 | 1.24 | 48 |
| `perf_486_486dx-33.log` | `486_target` | 2 | 40 | 32.08 | 1.25 | 510 |
| `perf_486_486dx-33.log` | `basic_runtime` | 4 | 33 | 27.62 | 1.19 | 313 |
| `perf_486_486dx2-66.log` | `real_inference` | 3 | 35 | 14.12 | 2.48 | 48 |
| `perf_486_486dx2-66.log` | `486_target` | 2 | 40 | 16.04 | 2.49 | 510 |
| `perf_486_486dx2-66.log` | `basic_runtime` | 4 | 33 | 13.84 | 2.38 | 313 |
| `perf_486_486dx2-66_kernel.log` | `real_inference` | 3 | 35 | 14.39 | 2.43 | 48 |
| `perf_486_486dx2-66_kernel.log` | `486_target` | 2 | 40 | 16.32 | 2.45 | 510 |
| `perf_486_486dx2-66_kernel.log` | `basic_runtime` | 4 | 33 | 14.06 | 2.35 | 313 |
| `perf_486_486dx2-66_model_headq4_prod_probe.log` | `real_inference` | 3 | 35 | 16.42 | 2.13 | 48 |
| `perf_486_486dx2-66_model_headq4_prod_probe.log` | `486_target` | 2 | 57 | 26.58 | 2.14 | 1245 |
| `perf_486_486dx2-66_model_headq4_prod_probe.log` | `basic_runtime` | 4 | 33 | 15.88 | 2.08 | 313 |
| `perf_486_486dx2-66_model_profile_386_min.log` | `real_inference` | 31 | 90 | 7.69 | 11.70 | 106 |
| `perf_486_486dx2-66_model_profile_386_min.log` | `486_target` | 19 | 90 | 6.87 | 13.10 | 34 |
| `perf_486_486dx2-66_model_profile_386_min.log` | `basic_runtime` | 27 | 90 | 7.41 | 12.15 | 103 |
| `perf_486_486dx2-66_model_profile_486dx2_usable.log` | `real_inference` | 31 | 90 | 34.00 | 2.65 | 34 |
| `perf_486_486dx2-66_model_profile_486dx2_usable.log` | `486_target` | 19 | 46 | 16.86 | 2.73 | 48 |
| `perf_486_486dx2-66_model_profile_486dx2_usable.log` | `basic_runtime` | 27 | 66 | 25.10 | 2.63 | 48 |
| `perf_486_486dx2-66_model_tokheadq4_prod_probe.log` | `real_inference` | 3 | 35 | 16.37 | 2.14 | 48 |
| `perf_486_486dx2-66_model_tokheadq4_prod_probe.log` | `486_target` | 2 | 33 | 15.38 | 2.15 | 65 |
| `perf_486_486dx2-66_model_tokheadq4_prod_probe.log` | `basic_runtime` | 4 | 49 | 23.40 | 2.09 | 313 |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe.log` | `real_inference` | 3 | 35 | 43.01 | 0.81 | 48 |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe.log` | `486_target` | 2 | 33 | 40.48 | 0.82 | 65 |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe.log` | `basic_runtime` | 4 | 49 | 60.69 | 0.81 | 313 |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `real_inference` | 3 | 35 | 43.34 | 0.81 | 48 |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `486_target` | 2 | 33 | 40.70 | 0.81 | 65 |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `basic_runtime` | 4 | 49 | 61.14 | 0.80 | 313 |
| `perf_486_486dx4-100.log` | `real_inference` | 3 | 35 | 7.09 | 4.94 | 48 |
| `perf_486_486dx4-100.log` | `486_target` | 2 | 40 | 7.97 | 5.02 | 510 |
| `perf_486_486dx4-100.log` | `basic_runtime` | 4 | 33 | 6.92 | 4.77 | 313 |
| `perf_486_486sx-25.log` | `real_inference` | 3 | 35 | 56.47 | 0.62 | 48 |
| `perf_486_486sx-25.log` | `486_target` | 2 | 40 | 64.10 | 0.62 | 510 |
| `perf_486_486sx-25.log` | `basic_runtime` | 4 | 33 | 55.31 | 0.60 | 313 |
| `perf_486_pentium-133.log` | `real_inference` | 3 | 35 | 3.51 | 9.97 | 48 |
| `perf_486_pentium-133.log` | `486_target` | 2 | 40 | 4.00 | 10.00 | 510 |
| `perf_486_pentium-133.log` | `basic_runtime` | 4 | 33 | 3.46 | 9.54 | 313 |
| `perf_486_pentium-60.log` | `real_inference` | 3 | 35 | 7.03 | 4.98 | 48 |
| `perf_486_pentium-60.log` | `486_target` | 2 | 40 | 8.02 | 4.99 | 510 |
| `perf_486_pentium-60.log` | `basic_runtime` | 4 | 33 | 6.92 | 4.77 | 313 |

## Kernel Stage Details

| Evidence | Stage | Calls | Seconds | Share |
|---|---|---:|---:|---:|
| `perf_486_486dx2-66_kernel.log` | `embedding` | 114 | 0.0000 | 0.0% |
| `perf_486_486dx2-66_kernel.log` | `ln1_qkv` | 228 | 2.3600 | 5.7% |
| `perf_486_486dx2-66_kernel.log` | `attention` | 228 | 0.9000 | 2.2% |
| `perf_486_486dx2-66_kernel.log` | `projection` | 228 | 0.9400 | 2.3% |
| `perf_486_486dx2-66_kernel.log` | `ffn` | 228 | 6.8000 | 16.3% |
| `perf_486_486dx2-66_kernel.log` | `final_head` | 108 | 30.7500 | 73.7% |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `embedding` | 123 | 0.0500 | 0.0% |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `ln1_qkv` | 246 | 2.5100 | 1.8% |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `attention` | 246 | 1.1400 | 0.8% |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `projection` | 246 | 1.4100 | 1.0% |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `ffn` | 246 | 6.9300 | 4.9% |
| `perf_486_486dx2-66_model_tokheadq4_stream_prod_probe_kernel.log` | `final_head` | 117 | 130.0300 | 91.5% |

## Runtime Contract

| Field | Value |
|---|---|
| Timed region | `decode_loop_only` |
| Sampling | `greedy_temperature_0` |
| Console progress | `disabled` |
| KV cache | `enabled` |
| Arithmetic | `q20.12_fixed` |

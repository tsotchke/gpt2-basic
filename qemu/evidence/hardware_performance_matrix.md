# GPT2-BASIC Physical Hardware Performance Matrix

This report is generated only from staged physical capture logs named
`hardware_<machine>_perf.log`. It deliberately does not read QEMU
`perf_486_*` logs, host-speed measurements, or planning estimates.

No staged physical hardware performance logs were found yet.

Run the DOS hardware capture on a real machine, copy the logs back,
then stage them with:

```sh
python3 scripts/stage_hardware_capture_evidence.py \
  --capture-dir /path/to/capture \
  --machine-key 486dx2_66_dos622
```

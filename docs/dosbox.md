# DOSBox Integration

The DOSBox bundle is a convenience path for people who want to run the current
preview without building a QEMU disk image or copying files to physical DOS
hardware. It uses the same `GPT2` DOS directory layout as the hardware-transfer
bundle, includes the `CWSDPMI.EXE` DPMI host required by the FreeBASIC-built
DOS executable, then adds DOSBox configuration files and host launchers.

Build it with:

```sh
python3 scripts/build_dosbox_bundle.py --force
```

The builder writes:

```text
/private/tmp/gpt2-basic-dosbox
/private/tmp/gpt2-basic-dosbox.zip
/private/tmp/gpt2-basic-dosbox.zip.sha256
```

## Running

Install DOSBox, DOSBox Staging, or DOSBox-X. From inside the extracted
`gpt2-basic-dosbox` directory:

```sh
dosbox -conf DOSBOX/GPT2MAN.CONF
dosbox -conf DOSBOX/GPT2CHAT.CONF
dosbox -conf DOSBOX/GPT2INT.CONF
dosbox -conf DOSBOX/GPT2DEMO.CONF
dosbox -conf DOSBOX/GPT2QUAL.CONF
dosbox -conf DOSBOX/GPT2PERF.CONF
```

On Unix-like hosts, the bundle also includes launchers:

```sh
./run-chat.sh
./run-demo.sh
./run-completion.sh
./run-quality.sh
./run-perf.sh
```

Set `DOSBOX=dosbox-x` or `DOSBOX=dosbox-staging` before running a shell launcher
if your binary has a different name.

On Windows, use the matching `RUN*.BAT` launchers from the extracted bundle.

## Profiles

| Config | Purpose | Output |
|---|---|---|
| `GPT2MAN.CONF` | Mount and leave DOSBox at `C:\GPT2`. | interactive DOS shell |
| `GPT2CHAT.CONF` | Run `ASSIST.EXE`. | pack-driven conversational assistant |
| `GPT2INT.CONF` | Run `GPT2.EXE` with no arguments. | prompt-based completion session |
| `GPT2DEMO.CONF` | Run the release demo. | screen output |
| `GPT2QUAL.CONF` | Run `GPT2.EXE --quality-all`. | `C:\GPT2\QUAL.LOG` |
| `GPT2PERF.CONF` | Run `GPT2.EXE --perf`. | `C:\GPT2\PERF.LOG` |
| `GPT2TRAC.CONF` | Run `GPT2.EXE --trace`. | `C:\GPT2\TRACE.LOG` |
| `GPT2SAMP.CONF` | Run `GPT2.EXE --sampling-matrix`. | `C:\GPT2\SAMPLE.LOG` |

Each config uses DOSBox's `[autoexec]` section to mount the extracted bundle as
drive `C:`, change to `C:\GPT2`, and optionally run one profile command. The
configs intentionally use `mount c .` so the zip can be extracted anywhere
without embedding a host-specific path.

Inside the manual profile, run `ASSIST.EXE` for the pack-driven chat shell, or
run `GPT2.EXE` with no arguments for raw prompt completion. The assistant shell
uses pack retrieval, golden replies, and guarded model output so prompt text and
repetition are not shown as chat answers.

## Scope

DOSBox integration is for quick demos, smoke tests, and user convenience. It is
not a substitute for the QEMU release gate or the physical 486 validation gate.
Use QEMU evidence for preview release claims and `docs/hardware-validation.md`
for real-machine evidence.

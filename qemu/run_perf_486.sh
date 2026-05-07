#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOT_IMAGE="$ROOT/qemu/boot-test.img"
HDD_IMAGE="$ROOT/qemu/gpt2hdd.img"

profile="${1:-486dx2-66}"
requested_model_dir="${2:-}"
perf_mode="${3:-perf}"

usage() {
    cat <<'USAGE'
Usage: bash qemu/run_perf_486.sh [profile] [model_dir] [perf|kernel]

Profiles:
  386dx-33     Conservative 386-class throttle, QEMU 486,-fpu CPU model, icount shift=7
  486sx-25     486 no-FPU throttle, QEMU 486,-fpu CPU model, icount shift=6
  486dx-33     486DX/33-era throttle, QEMU 486 CPU model, icount shift=5
  486dx2-66    486DX2/66-era throttle, QEMU 486 CPU model, icount shift=4
  486dx4-100   486DX4/100 upper-bound throttle, QEMU 486 CPU model, icount shift=3
  pentium-60   Pentium-era throttle, QEMU pentium CPU model, icount shift=3
  pentium-133  Pentium-era throttle, QEMU pentium CPU model, icount shift=2
  host         No icount throttle, useful as a fast emulator baseline

The DOS executable emits PERF_* records. The kernel mode also emits
KERNEL_PERF_* records from inside the fixed decode loop. QEMU icount is repeatable
instruction-count throttling, not cycle-accurate board emulation.
If model_dir is omitted, assets/gpt2_basic/MODEL is used.
USAGE
}

case "$perf_mode" in
    perf|kernel)
        ;;
    *)
        echo "unknown perf mode: $perf_mode" >&2
        usage >&2
        exit 1
        ;;
esac

cpu_model="486"
icount_shift=""
label=""

case "$profile" in
    386dx-33)
        icount_shift=7
        cpu_model="486,-fpu"
        label="386DX/33-class conservative throttle"
        ;;
    486sx-25)
        icount_shift=6
        cpu_model="486,-fpu"
        label="486SX/25 no-FPU throttle"
        ;;
    486dx-33)
        icount_shift=5
        label="486DX/33-era throttle"
        ;;
    486dx2-66)
        icount_shift=4
        label="486DX2/66-era throttle"
        ;;
    486dx4-100)
        icount_shift=3
        label="486DX4/100 upper-bound throttle"
        ;;
    pentium-60)
        icount_shift=3
        cpu_model="pentium"
        label="Pentium 60-era throttle"
        ;;
    pentium-133)
        icount_shift=2
        cpu_model="pentium"
        label="Pentium 133-era throttle"
        ;;
    host)
        label="host-speed emulator baseline"
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "unknown profile: $profile" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ ! -f "$HDD_IMAGE" ]]; then
    echo "missing $HDD_IMAGE" >&2
    echo "Create the FreeDOS/FreeBASIC hard-disk image first." >&2
    exit 1
fi

if [[ ! -f "$BOOT_IMAGE" ]]; then
    echo "missing $BOOT_IMAGE" >&2
    echo "Create the FreeDOS boot-floppy image first." >&2
    exit 1
fi

if pgrep -f qemu-system-i386 >/dev/null 2>&1; then
    echo "qemu-system-i386 is already running; stop it before updating the boot image." >&2
    exit 1
fi

MODEL_DIR=""
if [[ -n "$requested_model_dir" ]]; then
    if [[ "$requested_model_dir" == /* ]]; then
        MODEL_DIR="$requested_model_dir"
    else
        MODEL_DIR="$ROOT/$requested_model_dir"
    fi
elif [[ -d "$ROOT/assets/gpt2_basic/MODEL" ]]; then
    MODEL_DIR="$ROOT/assets/gpt2_basic/MODEL"
elif [[ -d "$ROOT/assets/tiny_gpt/MODEL" ]]; then
    MODEL_DIR="$ROOT/assets/tiny_gpt/MODEL"
fi

if [[ -n "$MODEL_DIR" ]]; then
    python3 "$ROOT/scripts/model_report.py" --model-dir "$MODEL_DIR" --strict
    python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
        --put-tree "$MODEL_DIR" MODEL
else
    echo "missing GPT2-BASIC MODEL directory" >&2
    echo "Run scripts/train_gpt2_basic.py on the host first." >&2
    exit 1
fi

model_name="$(basename "$MODEL_DIR")"
model_key="$(printf '%s' "$model_name" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | sed 's/^_*//;s/_*$//')"
if [[ "$model_key" == "model" ]]; then
    suffix=""
else
    suffix="_$model_key"
fi

if [[ "$perf_mode" == "kernel" ]]; then
    boot_auto="$ROOT/qemu/fdauto_kernel_perf.bat"
else
    boot_auto="$ROOT/qemu/fdauto_perf.bat"
fi

python3 "$ROOT/qemu/fat_image_put.py" "$BOOT_IMAGE" \
    --put "$boot_auto" FDAUTO.BAT

qemu_args=(
    -machine isapc
    -accel tcg
    -cpu "$cpu_model"
    -m 64
    -drive "file=$BOOT_IMAGE,format=raw,if=floppy,index=0"
    -drive "file=$HDD_IMAGE,format=raw,if=ide,index=0,media=disk"
    -boot a
    -display curses
    -monitor none
    -no-reboot
)

if [[ -n "$icount_shift" ]]; then
    qemu_args+=(
        -icount "shift=$icount_shift,sleep=on"
        -rtc clock=vm
    )
fi

echo "Running GPT2-BASIC performance suite under QEMU: $label"
echo "Performance mode: $perf_mode"
echo "QEMU CPU model: $cpu_model"
echo "Model directory: $MODEL_DIR"
if [[ -n "$icount_shift" ]]; then
    mips=$(awk "BEGIN { printf \"%.1f\", 1000 / (2 ^ $icount_shift) }")
    echo "Instruction-count target: about $mips million guest instructions/sec"
fi
echo "If the FreeDOS language menu appears, press Enter for English."

set +e
env TERM=xterm qemu-system-i386 "${qemu_args[@]}"
qemu_status=$?
set -e

if [[ "$qemu_status" -ne 0 && "$qemu_status" -ne 143 ]]; then
    exit "$qemu_status"
fi

mkdir -p "$ROOT/qemu/evidence"
if [[ "$perf_mode" == "kernel" ]]; then
    out_log="$ROOT/qemu/evidence/perf_486_${profile}${suffix}_kernel.log"
else
    out_log="$ROOT/qemu/evidence/perf_486_${profile}${suffix}.log"
fi
python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --get PERF.LOG "$out_log"

{
    echo "PERF_RUNNER|basis=qemu-emulation|profile=$profile|label=$label|model_dir=$MODEL_DIR|qemu_machine=isapc|qemu_cpu=$cpu_model|icount_shift=${icount_shift:-off}|accel=tcg|mode=$perf_mode"
} >> "$out_log"

if [[ -z "$suffix" && "$perf_mode" == "perf" && "$profile" == "486dx2-66" ]]; then
    cp "$out_log" "$ROOT/qemu/evidence/perf_486.log"
fi

report_inputs=(--input "$out_log")
for existing_log in "$ROOT"/qemu/evidence/perf_486_*.log; do
    if [[ -f "$existing_log" && "$existing_log" != "$out_log" ]]; then
        report_inputs+=(--input "$existing_log")
    fi
done

python3 "$ROOT/scripts/hardware_perf_report.py" \
    "${report_inputs[@]}" \
    --output "$ROOT/qemu/evidence/hardware_perf_report.md"

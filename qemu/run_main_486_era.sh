#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOT_IMAGE="$ROOT/qemu/boot-test.img"
HDD_IMAGE="$ROOT/qemu/gpt2hdd.img"

profile="${1:-486dx2-66}"

usage() {
    cat <<'USAGE'
Usage: bash qemu/run_main_486_era.sh [profile]

Profiles:
  386dx-33     Conservative 386-class throttle, QEMU 486 CPU model, icount shift=7
  486sx-25     486 no-FPU throttle, QEMU 486,-fpu CPU model, icount shift=6
  486dx-33     486DX/33-era throttle, QEMU 486 CPU model, icount shift=5
  486dx2-66    486DX2/66-era throttle, QEMU 486 CPU model, icount shift=4
  486dx4-100   486DX4/100 upper-bound throttle, QEMU 486 CPU model, icount shift=3
  pentium-60   Pentium-era throttle, QEMU pentium CPU model, icount shift=3
  pentium-133  Pentium-era throttle, QEMU pentium CPU model, icount shift=2
  host         No icount throttle, useful as the fast baseline

QEMU icount is instruction-count based, not cycle accurate. Treat these as
repeatable era-speed approximations, not exact timings for a real motherboard.
USAGE
}

cpu_model="486"

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
        icount_shift=""
        label="host-speed baseline"
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
if [[ -d "$ROOT/assets/gpt2_basic/MODEL" ]]; then
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

python3 "$ROOT/qemu/fat_image_put.py" "$BOOT_IMAGE" \
    --put "$ROOT/qemu/fdauto_run_main.bat" FDAUTO.BAT

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

echo "Running full GPT2-BASIC under QEMU: $label"
echo "QEMU CPU model: $cpu_model"
if [[ -n "$icount_shift" ]]; then
    mips=$(awk "BEGIN { printf \"%.1f\", 1000 / (2 ^ $icount_shift) }")
    echo "Instruction-count target: about $mips million guest instructions/sec"
fi
echo "If the FreeDOS language menu appears, press Enter for English."
echo "At the splash screen, press any key. Use menu option 1 for text completion."

exec env TERM=xterm qemu-system-i386 "${qemu_args[@]}"

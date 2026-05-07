#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HDD_IMAGE="$ROOT/qemu/gpt2hdd.img"
BOOT_IMAGE="$ROOT/qemu/boot-test.img"
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
    echo "qemu-system-i386 is already running; stop it before updating the disk images." >&2
    exit 1
fi

python3 "$ROOT/qemu/make_dos_staging.py"

python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --put-tree "$ROOT/qemu/staging/GPT2SRC" GPT2SRC
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
    --put "$ROOT/qemu/fdauto_compile.bat" FDAUTO.BAT

echo "Compiling full GPT2-BASIC under QEMU 486."
echo "If the FreeDOS language menu appears, press Enter for English."
echo "Expected success marker: COMPILE_OK"

set +e
env TERM=xterm qemu-system-i386 \
    -machine isapc \
    -cpu 486 \
    -m 64 \
    -drive "file=$BOOT_IMAGE,format=raw,if=floppy,index=0" \
    -drive "file=$HDD_IMAGE,format=raw,if=ide,index=0,media=disk" \
    -boot a \
    -display curses \
    -monitor none \
    -no-reboot
qemu_status=$?
set -e

if [[ "$qemu_status" -ne 0 && "$qemu_status" -ne 143 ]]; then
    exit "$qemu_status"
fi

mkdir -p "$ROOT/qemu/evidence"
python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --get COMPILE.LOG "$ROOT/qemu/evidence/compile_main_486.log" \
    --get GPT2.EXE "$ROOT/qemu/evidence/GPT2.EXE"

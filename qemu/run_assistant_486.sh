#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOT_IMAGE="$ROOT/qemu/boot-test.img"
HDD_IMAGE="$ROOT/qemu/gpt2hdd.img"
MODEL_DIR="${1:-$ROOT/assets/gpt2_basic/MODEL}"
PACK_DIR="${2:-$ROOT/assets/gpt2_basic/PACKS}"

if [[ "$MODEL_DIR" != /* ]]; then
    MODEL_DIR="$ROOT/$MODEL_DIR"
fi
if [[ "$PACK_DIR" != /* ]]; then
    PACK_DIR="$ROOT/$PACK_DIR"
fi

model_name="$(basename "$MODEL_DIR")"
model_key="$(printf '%s' "$model_name" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | sed 's/^_*//;s/_*$//')"
if [[ "$model_key" == "model" ]]; then
    suffix=""
else
    suffix="_$model_key"
fi

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

if [[ ! -d "$PACK_DIR" ]]; then
    echo "missing assistant pack directory: $PACK_DIR" >&2
    exit 1
fi

if pgrep -f qemu-system-i386 >/dev/null 2>&1; then
    echo "qemu-system-i386 is already running; stop it before updating the boot image." >&2
    exit 1
fi

python3 "$ROOT/qemu/make_dos_staging.py"

python3 "$ROOT/scripts/model_report.py" --model-dir "$MODEL_DIR" --strict
python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --put-tree "$ROOT/qemu/staging/GPT2SRC" GPT2SRC
python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --put-tree "$MODEL_DIR" MODEL
python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --put-tree "$PACK_DIR" PACKS
python3 "$ROOT/qemu/fat_image_put.py" "$BOOT_IMAGE" \
    --put "$ROOT/qemu/fdauto_assist.bat" FDAUTO.BAT

echo "Running GPT2-BASIC assistant pack shell under QEMU 486."
echo "Model directory: $MODEL_DIR"
echo "Pack directory: $PACK_DIR"
echo "If the FreeDOS language menu appears, press Enter for English."

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
assistant_log="$ROOT/qemu/evidence/assistant_486${suffix}.log"
compile_log="$ROOT/qemu/evidence/assistant_compile_486${suffix}.log"
python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --get ASSIST.LOG "$assistant_log" \
    --get ASSISTC.LOG "$compile_log"
echo "wrote $assistant_log"
echo "wrote $compile_log"

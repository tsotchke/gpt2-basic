#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOT_IMAGE="$ROOT/qemu/boot-test.img"
HDD_IMAGE="$ROOT/qemu/gpt2hdd.img"
MODEL_DIR="${1:-$ROOT/assets/gpt2_basic/MODEL}"
PACK_DIR="${2:-$ROOT/assets/gpt2_basic/PACKS}"
GPT2_EXE="${3:-$ROOT/qemu/evidence/GPT2.EXE}"
CAPTURE_DIR="$ROOT/qemu/evidence/hardware_capture_486_qemu"

if [[ "$MODEL_DIR" != /* ]]; then
    MODEL_DIR="$ROOT/$MODEL_DIR"
fi
if [[ "$PACK_DIR" != /* ]]; then
    PACK_DIR="$ROOT/$PACK_DIR"
fi
if [[ "$GPT2_EXE" != /* ]]; then
    GPT2_EXE="$ROOT/$GPT2_EXE"
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

if [[ ! -f "$GPT2_EXE" ]]; then
    echo "missing compiled GPT2.EXE: $GPT2_EXE" >&2
    echo "Run bash qemu/compile_main_486.sh first." >&2
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "missing model directory: $MODEL_DIR" >&2
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
    --remove GPT2 \
    --put "$GPT2_EXE" GPT2/GPT2.EXE \
    --put "$ROOT/hardware/HWVALID.BAT" GPT2/HWVALID.BAT \
    --put "$ROOT/hardware/HWNOTES.TXT" GPT2/HWNOTES.TXT

python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --put-tree "$MODEL_DIR" GPT2/MODEL \
    --put-tree "$PACK_DIR" GPT2/PACKS \
    --put-tree "$ROOT/qemu/staging/GPT2SRC" GPT2/GPT2SRC

python3 "$ROOT/qemu/fat_image_put.py" "$BOOT_IMAGE" \
    --put "$ROOT/qemu/fdauto_hwvalid.bat" FDAUTO.BAT

echo "Running GPT2-BASIC hardware-capture rehearsal under QEMU 486."
echo "Staged DOS path: C:\GPT2"
echo "Model directory: $MODEL_DIR"
echo "Pack directory: $PACK_DIR"
echo "Executable: $GPT2_EXE"
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

mkdir -p "$CAPTURE_DIR"
python3 "$ROOT/qemu/fat_image_put.py" "$HDD_IMAGE" \
    --get-text GPT2/HWVALID.LOG "$CAPTURE_DIR/HWVALID.LOG" \
    --get-text GPT2/QUAL.LOG "$CAPTURE_DIR/QUAL.LOG" \
    --get-text GPT2/PERF.LOG "$CAPTURE_DIR/PERF.LOG" \
    --get-text GPT2/ASSIST.LOG "$CAPTURE_DIR/ASSIST.LOG" \
    --get-text GPT2/ASSISTC.LOG "$CAPTURE_DIR/ASSISTC.LOG" \
    --get-text GPT2/HWNOTES.TXT "$CAPTURE_DIR/HWNOTES.TXT"

python3 "$ROOT/scripts/verify_hardware_capture.py" \
    --capture-dir "$CAPTURE_DIR" \
    > "$ROOT/qemu/evidence/hardware_capture_486_qemu_probe.log"

echo "wrote $CAPTURE_DIR"
echo "wrote $ROOT/qemu/evidence/hardware_capture_486_qemu_probe.log"

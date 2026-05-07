#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HDD_IMAGE="$ROOT/qemu/gpt2hdd.img"
BOOT_IMAGE="$ROOT/qemu/boot-test.img"
HDD_VOL="/Volumes/GPT2BASIC"
BOOT_VOL="/Volumes/FD14-BOOT"

if [[ ! -f "$HDD_IMAGE" ]]; then
    echo "missing $HDD_IMAGE" >&2
    echo "Run the FreeDOS/FreeBASIC image setup first." >&2
    exit 1
fi

if [[ ! -f "$BOOT_IMAGE" ]]; then
    echo "missing $BOOT_IMAGE" >&2
    echo "Run the FreeDOS/FreeBASIC image setup first." >&2
    exit 1
fi

if pgrep -f qemu-system-i386 >/dev/null; then
    echo "qemu-system-i386 is already running; stop it before updating the disk image." >&2
    exit 1
fi

python3 "$ROOT/qemu/make_dos_staging.py"

hdiutil attach "$HDD_IMAGE" >/dev/null
mkdir -p "$HDD_VOL/GPT2SRC"
COPYFILE_DISABLE=1 cp "$ROOT/qemu/staging/GPT2SRC/GPT2DOS.BAS" "$HDD_VOL/GPT2SRC/GPT2DOS.BAS"
hdiutil detach "$HDD_VOL" >/dev/null

hdiutil attach "$BOOT_IMAGE" >/dev/null
COPYFILE_DISABLE=1 cp "$ROOT/qemu/fdauto_run_gpt2dos.bat" "$BOOT_VOL/fdauto.bat"
hdiutil detach "$BOOT_VOL" >/dev/null

echo "Booting GPT2-BASIC DOS target under QEMU 486."
echo "After FreeDOS prints 'Disks stopped.', press Ctrl-C if QEMU remains open."

exec env TERM=xterm qemu-system-i386 \
    -machine isapc \
    -cpu 486 \
    -m 64 \
    -drive "file=$BOOT_IMAGE,format=raw,if=floppy,index=0" \
    -drive "file=$HDD_IMAGE,format=raw,if=ide,index=0,media=disk" \
    -boot a \
    -display curses \
    -monitor none \
    -no-reboot

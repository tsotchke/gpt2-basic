# Assistant Pack Generator Evidence

Date: 2026-05-21
Status: `PASS`

## Capability

`scripts/create_assistant_pack.py` creates a complete lightweight assistant
pack from a folder of ASCII notes.

Generated files:

- `PACK.INI`
- `HELP.TXT`
- `KNOW.TXT`
- `USER.TXT`
- `USAGE.TXT`
- `KDB.TXT`, `KDBIDX.TXT`, and `KDB?.TXT` text recall buckets
- `KB2ALL.BIN`, `KB2IDX.TXT`, and `KB2?.BIN` fixed-width binary recall pages
- Aggregate `KB2TERM.TXT` and sharded `KB2T?.TXT` term indexes

The default pack shares `PACKS\CHAT\MODEL`, `CHAT.SPR`, and `CHAT.ICN`, so a
new domain can behave like a language cartridge without adding another model
checkpoint. `--register` appends the pack ID to `PACKS.TXT`.

## Verification

```text
ASSISTANT_PACK_CREATE|pack=HWREPAIR|title=Hardware Repair|notes=1|know_rows=2|write=1|register=1
ASSISTANT_PACK_CREATED|pack=HWREPAIR|...|registered=1|rows=5
PROBE_OK assistant_pack_create=1
PROBE_OK assistant_pack_create_self_test=1
```

The self-test builds a temporary `HWREPAIR` pack, registers it, validates it
with the shared pack contract, and confirms both the generated `KB2TERM.TXT`
file and at least one `KB2T?.TXT` shard exist.

`PORTABLE` is the first shipped pack generated through this workflow. Its
source notes live in `data/assistant_pack_notes/portable`, it shares
`PACKS\CHAT\MODEL`, and it contributes six additional retrieval/usefulness/KDB
cases to the assistant harness.

## Operator Command

```sh
python3 scripts/create_assistant_pack.py --pack HWREPAIR --title "Hardware Repair" --notes-dir notes/hardware --write --register
python3 scripts/validate_assistant_pack_authoring.py
```

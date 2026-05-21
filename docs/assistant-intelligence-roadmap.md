# Assistant Intelligence Roadmap

GPT2-BASIC should feel more modern than its model size suggests by treating
the 486 as a local agent computer, not just a neural network host.

## Current Architecture

- Pack-local model weights are hot-loadable with `/pack NAME`.
- `HELP.TXT` and `KNOW.TXT` are human-editable authoring files.
- `KDB.TXT` is the generated readable recall surface, with `KDBIDX.TXT` and
  `KDB?.TXT` bucket files as text fallback.
- `KB2ALL.BIN`, `KB2IDX.TXT`, and `KB2?.BIN` are the fixed-width compiled
  recall pages used by the DOS fast path.
- `USER.TXT` is a local override file users can edit on the target machine.
- `ASSIST.MEM` persists core memory facts across interactive sessions.
- `DEV` and `PORTABLE` demonstrate domain packs that reuse CHAT weights with
  their own KDB/KB2 recall surfaces.
- `scripts/create_assistant_pack.py` creates a complete lightweight pack from
  a folder of ASCII notes, including generated KB2 binary recall, aggregate
  term indexes, and sharded term indexes.
- Golden replies, retrieval, memory, and generation are all reported in
  `ASSIST_REPLY` evidence records.

See [`substrate-portability.md`](substrate-portability.md) for the minimum
runtime-primitives argument behind the BASIC implementation and future C,
assembly, and Eshkol ports.

## Design Direction

The target experience is a cartridge-like language system:

1. Keep one small resident model for general language behavior.
2. Hot-load pack weights only when a domain needs a different style.
3. Move facts, procedures, examples, and local preferences into pack files.
4. Compile authored notes into compact indexed text databases, bucket shards,
   and fixed-width binary pages.
5. Let user notes and persistent memory override bundled knowledge.
6. Treat retrieval as the first-class intelligence layer, with generation as
   a short synthesis layer.

## Storage Strategy

The DOS runtime should avoid scanning large prose files when possible.
`KDB.TXT` stores compact terms next to the answer text as the readable source
of generated recall. `KB2ALL.BIN` and `KB2?.BIN` store the same rows as
fixed-width binary records, so the shell can scan bounded records without
line parsing. `KB2T?.TXT` shards map significant terms to likely KB2 row IDs
before the shell opens binary records. `KDBIDX.TXT` and `KB2IDX.TXT` list
generated bucket shards, and the shell scans only the buckets suggested by
significant query words before falling back to the full KDB.

## Completed Milestones

- Pack generator from a folder of notes: `scripts/create_assistant_pack.py`
  writes `PACK.INI`, authoring files, `USER.TXT`, `USAGE.TXT`, generated
  `KDB.TXT` buckets, compiled `KB2*.BIN` pages, aggregate `KB2TERM.TXT`, and
  sharded `KB2T?.TXT` term indexes, while sharing `PACKS\CHAT\MODEL` by
  default.
- Lightweight domain pack without retraining: `PORTABLE` ships portable
  intelligence notes generated from `data/assistant_pack_notes/portable` and
  shares the CHAT model.
- Retrieval-only recall probe: `ASSIST.EXE --recall-probe` measures the KB2/KDB
  recall path across every shipped pack without model generation.

## Next Milestones

- Add more domain packs for hardware repair, programming, and offline reference
  manuals using the same generated KDB/KB2 contract.
- Compare recall-probe timing on QEMU and physical hardware, then decide
  whether the next storage step should be topic shards or offset tables.
- Add persistent memory slots beyond name, goal, style, and problem.
- Add a measured recall benchmark in QEMU and on physical 486 hardware.

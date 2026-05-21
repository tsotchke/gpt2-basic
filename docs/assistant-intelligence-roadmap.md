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

## Current Evidence Baseline

- CHAT pack quality: `PASS 160/160`, average `0.999`.
- Raw direct prompt gate: `PASS 83/83`.
- Generalist conversational prompt gate: `PASS 24/24`.
- Consistency gate: `PASS 498/498` variants across `83/83` groups.
- Pack retrieval, KDB index, KDB binary, and KB2 term-index gates: `PASS 42/42`.
- Usefulness workflow gate: `PASS 37/37` tasks across `9/9` workflows.
- QEMU assistant stress gate: `PASS 50/50` replies across five packs.
- Runtime bundle verification: preview, DOSBox, hardware-transfer, and launch-kit
  zips rebuild with sidecar checksums and without host-only training corpora.

Physical 486-class validation remains outside this roadmap pass and is tracked
separately in public issues #1 and #2.

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

1. Build a denser binary term index for local recall:
   https://github.com/tsotchke/gpt2-basic/issues/37
2. Add latency and recall-budget gates to the assistant harness:
   https://github.com/tsotchke/gpt2-basic/issues/41
3. Add an inspectable persistent conversation database:
   https://github.com/tsotchke/gpt2-basic/issues/39
4. Add a pack intent router for assistant queries:
   https://github.com/tsotchke/gpt2-basic/issues/38
5. Add larger programming and offline-reference domain packs:
   https://github.com/tsotchke/gpt2-basic/issues/40
6. Define Windows 95 and OS/2 assistant shell parity:
   https://github.com/tsotchke/gpt2-basic/issues/42
7. Repair and promote smaller model profiles under strict gates:
   https://github.com/tsotchke/gpt2-basic/issues/43
8. Prototype a C or assembly fixed-point kernel compatibility path:
   https://github.com/tsotchke/gpt2-basic/issues/44

The ordering is deliberate: recall density and latency budgets should come
before larger packs, and persistent memory plus routing should land before
native shell ports so every shell can target the same behavior.

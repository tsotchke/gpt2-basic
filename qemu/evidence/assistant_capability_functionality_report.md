# Assistant Capability And Functionality Report

Date: 2026-05-21
Status: `PASS`

This report is generated from repository evidence files by `scripts/build_assistant_capability_report.py`.

## Runtime Capability

- Runs under FreeDOS/QEMU 486 with 5 assistant packs: `CHAT`, `DOSHELP`, `OFFICE`, `DEV`, `PORTABLE`.
- Supports hot pack switching through `PACKS.TXT` and each pack's `PACK.INI`.
- Supports pack-local model paths, pack-local art assets, pack-local golden rows, pack-local help/knowledge rows, and editable `USER.TXT` notes.
- Uses retrieval-first answering before model synthesis: golden rows, compiled knowledge recall, session memory, and fallback checks are explicit in `ASSIST_REPLY`.
- Reports structured provenance and timing for every reply: `source`, `recall`, `recall_score`, `t_retrieve_ms`, `t_golden_ms`, `t_memory_ms`, `t_model_ms`, and `t_total_ms`.
- Interactive shell exposes `/capabilities`, `/limits`, `/sources`, `/status`, `/about`, `/pack`, `/memory`, `/remember KEY=VALUE`, and `/forget`.

## Recall And Storage

- Text KDB remains the readable source/fallback format: `KDB.TXT`, `KDBIDX.TXT`, and `KDB?.TXT`.
- Compiled KB2 recall ships for each pack: `KB2ALL.BIN`, `KB2IDX.TXT`, `KB2?.BIN`, and `KB2TERM.TXT`.
- KB2 files use fixed-width records for 486-friendly sequential reads and avoid reparsing large text rows during recall.
- `KB2TERM.TXT` is a compact per-pack inverted term index. The DOS runtime scores likely row IDs first, then falls back to binary buckets and finally text KDB recall.
- Current compiled KB2 payload sizes:
  - `CHAT`: 78 rows, 23 buckets, 159616 binary bytes, 4280 term-index bytes.
  - `DOSHELP`: 26 rows, 21 buckets, 55488 binary bytes, 2193 term-index bytes.
  - `OFFICE`: 27 rows, 20 buckets, 57504 binary bytes, 2458 term-index bytes.
  - `DEV`: 23 rows, 23 buckets, 49376 binary bytes, 2375 term-index bytes.
  - `PORTABLE`: 11 rows, 16 buckets, 23968 binary bytes, 1292 term-index bytes.
- Binary recall evaluation: `PASS 42/42`.
- Binary candidate row scan ratio: `0.531`.
- Binary candidate byte ratio: `0.688`.
- Term-index recall evaluation: `PASS 42/42`.
- Term-index candidate row scan ratio: `0.145`.
- Term-index candidate byte ratio: `0.315`.

## Language Coverage

- Raw direct model prompt gate: `PASS 83/83`.
- Generalist conversational prompt gate: `PASS 24/24`.
- Consistency gate: `PASS 498/498 variants, 83/83 groups`.
- Pack retrieval gate: `PASS 42/42`.
- Usefulness workflow gate: `PASS 37/37 tasks, 9/9 workflows`.
- KDB text index gate: `PASS 42/42`.
- KDB binary gate: `PASS 42/42`.
- KDB term-index gate: `PASS 42/42`.

Covered categories include general chat, identity, local inference, offline limits, prompt repair, repeated-answer recovery, troubleshooting, DOS setup, office writing, developer pack authoring, and portable-intelligence concepts.

Usefulness workflows currently cover operator prompts, trust/offline limits, DOS setup and repair, hardware transfer and emulator evidence, office handoffs, planning and risk, developer pack authoring, fast local recall architecture, and portable intelligence.

## DOS/QEMU Stress Result

- Scripted QEMU assistant run: `PASS`, reached `ASSIST_END|packs=5`.
- Stress QEMU run: `PASS`, reached `ASSIST_END|suite=stress-probe|packs=5`.
- Stress replies: `50`.
- Stress source mix: `golden=26 retrieval=16 model=0 fallback=0 memory=8`.
- Average total reply time in the stress report: `134 ms`.
- Average retrieval time in the stress report: `80 ms`.
- Recall modes in the stress report: `kb2_bucket=3 kb2_term=46 none=1`.
- Visible-answer validation: `PASS`.

## Hardware-Capture Rehearsal

- QEMU rehearses the physical `C:\GPT2\HWVALID.BAT` path before real transfer.
- Hardware-capture rehearsal: `PASS`.
- Hardware-capture assistant stress replies: `50`.
- Hardware-capture stress source mix: `golden=26 retrieval=16 model=0 fallback=0 memory=8`.
- Hardware-capture average total reply time: `28 ms`.
- Hardware-capture average retrieval time: `24 ms`.
- Physical machine capture status: PENDING: no staged physical `hardware_<machine>_manifest.md` capture is present yet.

## Authoring And Import

- `scripts/import_assistant_notes.py` can import ASCII notes into `USER.TXT` or `KNOW.TXT`.
- `--target user` writes machine-local notes without changing bundled pack knowledge.
- `--target know --rebuild-kdb` updates bundled pack knowledge and regenerates KDB/KB2 artifacts.
- `scripts/create_assistant_pack.py` can create a complete lightweight pack from a folder of ASCII notes, sharing `PACKS\CHAT\MODEL` by default.
- The pack generator writes `PACK.INI`, authoring files, `USER.TXT`, `USAGE.TXT`, generated KDB buckets, compiled KB2 pages, and `KB2TERM.TXT`.
- Authoring validator checks required pack files, source rows, generated text KDB, generated binary KDB, and model references.

## Release Payload

- Preview package manifest: `included`.
- Preview release tracked-input gate: `PASS`.
- Preview artifact verifier: `PASS`.
- Release sidecar hashes: `PASS`.
- Runtime bundles exclude host-only `TRAIN.TXT` and `TOKBASE.TXT`.

## Known Limits

- This is not a frontier-scale LLM. It is a retrieval-first, pack-specialized DOS assistant with a very small local model.
- The strongest behavior comes from curated pack knowledge, golden rows, session memory, and fast local recall.
- Long, ambiguous, or out-of-domain prompts should be shortened or moved into an appropriate pack.
- No live web, news, package registry, or network lookup is available inside DOS.
- Current 486 stress replies did not require raw model generation; that is intentional for reliability and speed on this hardware class.
- Physical 486-class board evidence is still pending until real hardware returns the `HWVALID.LOG`, `QUAL.LOG`, `PERF.LOG`, `ASSIST.LOG`, `ASTRESS.LOG`, `ASSISTC.LOG`, and `HWNOTES.TXT` set.

## Next Production Targets

- Convert `KB2TERM.TXT` into an even denser binary term index once the text format has stabilized under real authoring changes.
- Add larger domain packs with the same KB2 contract, especially hardware repair, programming, office workflows, and offline reference manuals.
- Add a compact on-disk conversation database so memory persists across sessions while remaining inspectable and editable.
- Add a pack-selection router so the shell can recommend or switch packs from query intent.
- Add latency budgets per pack and fail the harness if retrieval or total reply time regresses beyond the 486 profile target.

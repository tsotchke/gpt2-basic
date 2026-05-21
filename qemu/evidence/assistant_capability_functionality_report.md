# Assistant Capability And Functionality Report

Date: 2026-05-20
Status: `PASS`

## Runtime Capability

- Runs under FreeDOS/QEMU 486 with four assistant packs: `CHAT`, `DOSHELP`, `OFFICE`, and `DEV`.
- Supports hot pack switching through `PACKS.TXT` and each pack's `PACK.INI`.
- Supports pack-local model paths, pack-local art assets, pack-local golden rows, pack-local help/knowledge rows, and editable `USER.TXT` notes.
- Uses retrieval-first answering before model synthesis: golden rows, compiled knowledge recall, session memory, and fallback checks are all explicit in `ASSIST_REPLY`.
- Reports structured provenance and timing for every reply: `source`, `recall`, `recall_score`, `t_retrieve_ms`, `t_golden_ms`, `t_memory_ms`, `t_model_ms`, and `t_total_ms`.
- Interactive shell exposes `/capabilities`, `/limits`, `/sources`, `/status`, `/about`, and `/pack`.
- The answer display now includes a compact source line such as `Source: golden / kdb_text_bucket ( 110 ms)`.

## Recall And Storage

- Text KDB remains the readable source/fallback format: `KDB.TXT`, `KDBIDX.TXT`, and `KDB?.TXT`.
- New compiled KB2 recall is shipped for each pack: `KB2ALL.BIN`, `KB2IDX.TXT`, and `KB2?.BIN`.
- KB2 files use fixed-width records for 486-friendly sequential reads and avoid reparsing large text rows during recall.
- Current compiled KB2 payload sizes:
  - `CHAT`: 78 rows, 23 buckets, 159616 bytes across full and bucket files.
  - `DOSHELP`: 26 rows, 21 buckets, 55488 bytes.
  - `OFFICE`: 27 rows, 20 buckets, 57504 bytes.
  - `DEV`: 23 rows, 23 buckets, 49376 bytes.
- Binary recall evaluation: `PASS 36/36`.
- Binary candidate row scan ratio: `0.524`.
- Binary candidate byte ratio: `0.669`.

## Language Coverage

- Raw direct model prompt gate: `PASS 83/83`.
- Generalist conversational prompt gate: `PASS 24/24`.
- Consistency gate: `PASS 498/498 variants, 83/83 groups`.
- Pack retrieval gate: `PASS 36/36`.
- Usefulness workflow gate: `PASS 31/31 tasks, 8/8 workflows`.
- KDB text index gate: `PASS 36/36`.
- KDB binary gate: `PASS 36/36`.

Covered categories include:

- General chat, identity, local inference, local limits, offline/no-web behavior, prompt quality, repeated-answer recovery, confidence framing, simple explanation, and lightweight planning.
- Troubleshooting, debugging, release checks, DPMI/CWSDPMI, CONFIG.SYS, AUTOEXEC.BAT, FAT image limits, QEMU logs, and real-hardware copy preparation.
- Rewriting, summarizing, shortening, release notes, status updates, handoff notes, bug reports, meeting notes, risk registers, project plans, customer replies, and user docs.
- Developer-pack guidance for retrieval-first design, authoring packs, fast recall storage, release checks, failure records, and modern 486 assistant architecture.

Usefulness workflows currently cover operator prompts, trust/offline limits, DOS
setup and repair, hardware transfer and emulator evidence, office handoffs,
planning and risk, developer pack authoring, and fast local recall architecture.

## DOS/QEMU Stress Result

- Scripted QEMU assistant run: `PASS`, reached `ASSIST_END|packs=4`.
- Stress QEMU run: `PASS`, reached `ASSIST_END|suite=stress-probe|packs=4`.
- Stress replies: `44`.
- Stress source mix: `golden=26 retrieval=10 model=0 fallback=0 memory=8`.
- Average total reply time in the stress report: `150 ms`.
- Average retrieval time in the stress report: `125 ms`.
- Recall modes in the stress report: `kdb_text_bucket=43 none=1`.
- Visible-answer validation: `PASS`.

## Authoring And Import

- `scripts/import_assistant_notes.py` can import ASCII notes into `USER.TXT` or `KNOW.TXT`.
- Import is dry-run by default.
- `--target user` writes machine-local notes without changing bundled pack knowledge.
- `--target know --rebuild-kdb` updates bundled pack knowledge and regenerates KDB/KB2 artifacts.
- Authoring validator checks required pack files, source rows, generated text KDB, generated binary KDB, and model references.

## Release Payload

- Preview release tracked-input gate: `PASS`.
- Preview artifact verifier: `PASS`.
- DOSBox zip unzip test: `PASS`.
- Launch-kit zip unzip test: `PASS`.
- Release sidecar hashes: `PASS`.
- Runtime bundles exclude host-only `TRAIN.TXT` and `TOKBASE.TXT`.

## Known Limits

- This is not a frontier-scale LLM. It is a retrieval-first, pack-specialized DOS assistant with a very small local model.
- The strongest behavior comes from curated pack knowledge, golden rows, session memory, and fast local recall.
- Long, ambiguous, or out-of-domain prompts should be shortened or moved into an appropriate pack.
- No live web, news, package registry, or network lookup is available inside DOS.
- Current 486 stress replies did not require raw model generation; that is intentional for reliability and speed on this hardware class.

## Next Production Targets

- Add a true KB2 binary fast path marker in the DOS stress report once runtime instrumentation can distinguish binary bucket hits from text bucket fallback at the report level.
- Add larger domain packs with the same KB2 contract, especially hardware repair, programming, office workflows, and offline reference manuals.
- Add a compact on-disk conversation database so memory persists across sessions while remaining inspectable and editable.
- Add a pack-selection router so the shell can recommend or switch packs from query intent.
- Add latency budgets per pack and fail the harness if retrieval or total reply time regresses beyond the 486 profile target.

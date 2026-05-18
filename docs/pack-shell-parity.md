# Assistant Pack Shell Parity Plan

The DOS assistant shell is the release baseline. Windows and OS/2 shells must
reuse the same pack metadata contract instead of inventing shell-specific pack
formats.

## Shared Pack Contract

The host contract lives in `scripts/assistant_pack_contract.py`. It is a
validator and reference parser for every shell implementation.

Pack root:

```text
assets/gpt2_basic/PACKS/
  PACKS.TXT
  CHAT/
    PACK.INI
    HELP.TXT
    USAGE.TXT
    CHAT.SPR
    CHAT.ICN
    MODEL/
```

`PACKS.TXT` rules:

- ASCII text only.
- Blank lines and lines starting with `#` or `;` are ignored.
- Pack IDs are uppercase 8.3-safe directory names: `A-Z`, `0-9`, `_`, up to 8
  characters.
- The first listed pack is the default. For the preview release that remains
  `CHAT`.
- Duplicate pack IDs are invalid.

`PACK.INI` required keys:

```ini
ID=CHAT
TITLE=Conversation Pack
MODEL=PACKS\CHAT\MODEL
PERSONA=You are a small friendly DOS chat assistant. Keep replies short.
HELP=HELP.TXT
USAGE=USAGE.TXT
SPRITE=CHAT.SPR
ICONS=CHAT.ICN
ACTIONS=chat,ask,idea,explain,cancel
```

Key rules:

- Keys are case-insensitive.
- Values are ASCII.
- `ID` must match the pack directory and `PACKS.TXT` entry.
- `MODEL` should point at a pack-local model directory with `PACKS\<ID>\MODEL`.
- `HELP`, `USAGE`, `SPRITE`, and `ICONS` should be pack-local filenames unless
  there is a documented reason to use a `PACKS\...` path.
- Path traversal with `..` is invalid.
- `ACTIONS` is a comma-separated list of lowercase shell tokens and must
  include `cancel`.

`HELP.TXT` rules:

- ASCII text only.
- Blank lines and comment lines are ignored.
- Each retrieval row is `key|title|text`.
- Every pack must have at least one retrieval row.

`USAGE.TXT` rules:

- ASCII text only.
- Each shell must be able to show this text through its pack help command.
- Required section markers:
  - `Purpose:`
  - `How it works:`
  - `How to use it:`
  - `Good prompts:`
  - `Actions:`

Model directory rules:

- Pack-local models must include `GPT2CFG.TXT`, `GPT2WT.BIN`, `GPT2FX.BIN`,
  `GPT2EXP.BIN`, `PROFILE.TXT`, and `VOCAB.BIN`.
- A shell may use extra optimized artifacts when present, but missing optional
  artifacts must not break loading.

Sprite/icon rules:

- `SPRITE` and `ICONS` are required metadata fields even when a shell renders
  a text-only UI.
- DOS currently reserves these assets and renders a text bubble/action UI.
- Windows and OS/2 may render native artwork later, but pack selection and
  usage behavior must not depend on artwork support.

## DOS Baseline

The DOS baseline is `src/assistant.bas` and `bash qemu/run_assistant_486.sh`.
It already emits structured evidence:

```text
ASSIST_BEGIN|suite=pack-shell|version=1
ASSIST_PACK|id=CHAT|...
ASSIST_MODEL|pack=CHAT|...
ASSIST_REPLY|pack=CHAT|...
ASSIST_END|packs=3
```

Any future shell should produce equivalent shell-specific records so parity is
auditable without screenshots.

## Windows Probe Plan

Before a Windows shell counts as release scope, add a scripted probe that:

1. Starts from the same `assets/gpt2_basic/PACKS` tree.
2. Loads `PACKS.TXT` and confirms `CHAT`, `DOSHELP`, and `OFFICE` in that
   order.
3. Parses each `PACK.INI` through the shared contract.
4. Prints each pack title and `USAGE.TXT`.
5. Resolves each pack-local `MODEL=PACKS\<ID>\MODEL`.
6. Emits structured records:

```text
WIN_ASSIST_BEGIN|suite=pack-shell|version=1
WIN_ASSIST_PACK|id=CHAT|title=Conversation Pack|model=PACKS\CHAT\MODEL
WIN_ASSIST_MODEL|pack=CHAT|path=PACKS\CHAT\MODEL|profile=...
WIN_ASSIST_USAGE|pack=CHAT|sections=5
WIN_ASSIST_END|packs=3
```

The first Windows release gate is parser and model-path parity. Native window
chrome, artwork, and text-entry polish can follow after the contract is proven.

## OS/2 Probe Plan

Before an OS/2 shell counts as release scope, add a scripted probe that:

1. Uses the same pack tree and the same `PACK.INI` semantics.
2. Verifies `HELP.TXT` retrieval rows are readable under OS/2 filesystem rules.
3. Shows `/about` or equivalent pack instructions from `USAGE.TXT`.
4. Resolves pack-local model paths without rewriting the pack metadata.
5. Emits structured records:

```text
OS2_ASSIST_BEGIN|suite=pack-shell|version=1
OS2_ASSIST_PACK|id=CHAT|title=Conversation Pack|model=PACKS\CHAT\MODEL
OS2_ASSIST_MODEL|pack=CHAT|path=PACKS\CHAT\MODEL|profile=...
OS2_ASSIST_USAGE|pack=CHAT|sections=5
OS2_ASSIST_END|packs=3
```

The OS/2/Warp package remains a later release track. The first useful OS/2
milestone is pack metadata parity, not a full native UI.

## Verification

Run the shared contract test and the existing DOS assistant verifier:

```sh
python3 scripts/assistant_pack_contract.py --self-test
python3 -m unittest discover -s tests -p 'test_assistant_pack_contract.py'
python3 scripts/verify_assistant_packs.py
```

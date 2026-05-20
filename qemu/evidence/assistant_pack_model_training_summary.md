# Assistant Pack Model Training Summary

| Pack | Model | Quality | Pass Rate | Train Log | Model Report | Quality Report |
|---|---|---:|---:|---|---|---|
| CHAT | assets/gpt2_basic/PACKS/CHAT/MODEL | 0.999 | 160/160 | qemu/evidence/train_assistant_chat.log | qemu/evidence/model_report_assistant_chat.log | qemu/evidence/quality_report_assistant_chat.md |
| DOSHELP | assets/gpt2_basic/PACKS/DOSHELP/MODEL | 1.000 | 4/4 | qemu/evidence/train_assistant_doshelp.log | qemu/evidence/model_report_assistant_doshelp.log | qemu/evidence/quality_report_assistant_doshelp.md |
| OFFICE | assets/gpt2_basic/PACKS/OFFICE/MODEL | 1.000 | 4/4 | qemu/evidence/train_assistant_office.log | qemu/evidence/model_report_assistant_office.log | qemu/evidence/quality_report_assistant_office.md |

Pack-local `MODEL=` paths are updated to `PACKS\<ID>\MODEL` so DOS, Windows, and OS/2 shells can share the same pack metadata.
